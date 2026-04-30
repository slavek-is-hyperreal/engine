// -----------------------------------------------------------------------------
// Plik: rtl/eml_dot_product_64.v
// Opis: Akcelerator iloczynu skalarnego bazujący na zbalansowanym drzewie
// turniejowym (Kogge-Stone) dedykowany architekturze EML (z wagami ASIS).
// POPRAWKI: 
// - D.4.2: Naprawiono wejścia węzła korzenia (root node).
// - D.4.3: Naprawiono przypisanie valid_out.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module eml_dot_product_64 (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_in,
    input  wire [31:0] x  [0:63],  // Aktywacje (ciąg wejściowy)
    input  wire [31:0] w  [0:63],  // Wagi pre-negowane ASIS (W[1..63] < 0)
    output wire [31:0] result,
    output wire        valid_out
);

    // Definicja zmiennej konfiguracyjnej i iteratora blokowego
    genvar i;

    // =========================================================================
    // WARSTWA 0: Faza Ekstrakcji - Mnożenie 64 Równoległych Zmiennych (3 cykle)
    // =========================================================================
    wire [31:0] mul_out [0:63];
    reg  [31:0] l0_reg  [0:63];

    // Rejestry propagujące status sygnału dla 3 cykli IP mnożnika
    reg valid_mul1, valid_mul2, valid_mul3;

    generate
        for (i = 0; i < 64; i = i + 1) begin : MUL_LAYER
            // Powszechne IP zmiennoprzecinkowe traktujące wagi z Constant Folding
            float_mul fmul_inst (
               .a(x[i]),
               .b(w[i]),
               .out(mul_out[i])
            );
            always @(posedge clk) begin
                l0_reg[i] <= mul_out[i];
            end
        end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_mul1 <= 0; valid_mul2 <= 0; valid_mul3 <= 0;
        end else begin
            valid_mul1 <= valid_in;
            valid_mul2 <= valid_mul1;
            valid_mul3 <= valid_mul2;
        end
    end

    // =========================================================================
    // WARSTWY 1-6: Drzewo Redukcyjne Subtrakcji (12 cykli retimingu)
    // Ze względu na specyfikację RTL ASIS wykorzystano bezpośrednie odjęcie
    // float_sub w miejsce redundancji pełnej komórki EML.
    // =========================================================================
    reg [31:0] l1_reg [0:31];
    reg [31:0] l2_reg [0:15];
    reg [31:0] l3_reg [0:7];
    reg [31:0] l4_reg [0:3];
    reg [31:0] l5_reg [0:1];
    reg [31:0] l6_reg;

    generate
        // WARSTWA 1 (Głębokość 1): 64 -> 32 węzły
        for (i = 0; i < 32; i = i + 1) begin : L1_SUB
            wire [31:0] sub_w;
            float_sub fsub_1 (.a(l0_reg[2*i]),.b(l0_reg[2*i+1]),.out(sub_w));
            always @(posedge clk) l1_reg[i] <= sub_w;
        end

        // WARSTWA 2 (Głębokość 2): 32 -> 16 węzłów
        for (i = 0; i < 16; i = i + 1) begin : L2_SUB
            wire [31:0] sub_w;
            float_sub fsub_2 (.a(l1_reg[2*i]),.b(l1_reg[2*i+1]),.out(sub_w));
            always @(posedge clk) l2_reg[i] <= sub_w;
        end

        // WARSTWA 3 (Głębokość 3): 16 -> 8 węzłów
        for (i = 0; i < 8; i = i + 1) begin : L3_SUB
            wire [31:0] sub_w;
            float_sub fsub_3 (.a(l2_reg[2*i]),.b(l2_reg[2*i+1]),.out(sub_w));
            always @(posedge clk) l3_reg[i] <= sub_w;
        end

        // WARSTWA 4 (Głębokość 4): 8 -> 4 węzły
        for (i = 0; i < 4; i = i + 1) begin : L4_SUB
            wire [31:0] sub_w;
            float_sub fsub_4 (.a(l3_reg[2*i]),.b(l3_reg[2*i+1]),.out(sub_w));
            always @(posedge clk) l4_reg[i] <= sub_w;
        end

        // WARSTWA 5 (Głębokość 5): 4 -> 2 węzły
        for (i = 0; i < 2; i = i + 1) begin : L5_SUB
            wire [31:0] sub_w;
            float_sub fsub_5 (.a(l4_reg[2*i]),.b(l4_reg[2*i+1]),.out(sub_w));
            always @(posedge clk) l5_reg[i] <= sub_w;
        end

        // WARSTWA 6 (Głębokość 6): 2 -> 1 węzeł bazowy (Korzeń Drzewa)
        wire [31:0] root_w;
        // POPRAWKA D.4.2: Wejścia l5_reg[0] i l5_reg[1] zamiast dwukrotnie l5_reg
        float_sub fsub_6 (.a(l5_reg[0]),.b(l5_reg[1]),.out(root_w));
        always @(posedge clk) l6_reg <= root_w;
    endgenerate

    // Propagacja opóźnienia dla potoku drzewnego (łącznie 12 cykli)
    reg [11:0] valid_tree_shift;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_tree_shift <= 12'd0;
        end else begin
            valid_tree_shift <= {valid_tree_shift[10:0], valid_mul3};
        end
    end

    // Mapowanie wyników
    assign result = l6_reg;
    // POPRAWKA D.4.3: Wybranie ostatniego bitu z rejestru przesuwnego
    assign valid_out = valid_tree_shift[11];

endmodule
