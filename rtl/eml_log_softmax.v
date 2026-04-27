// -----------------------------------------------------------------------------
// Plik: rtl/eml_log_softmax.v
// Opis: Akceleracja sprzętowa Log-Softmax jako natywnej operacji EML.
// Czasowa synchronizacja zachodzi przez linię opóźniającą sygnały wejściowe.
// POPRAWKA: D.4.4: Naprawiono wejścia sumatora korzenia (root adder).
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module eml_log_softmax #(
    parameter N = 8  // Rozmiar analizowanego wektora
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_in,
    input  wire [31:0] x [0:N-1],
    output wire [31:0] y [0:N-1],  // Wynik log_softmax(x_i)
    output wire        valid_out
);

    genvar i;

    // =========================================================================
    // KROK 1: Obliczenie exp(x_j) (współużytkowana logika fast_exp)
    // =========================================================================
    wire [31:0] exp_out [0:N-1];
    reg  [31:0] exp_reg [0:N-1];
    
    generate
        for (i = 0; i < N; i = i + 1) begin : EXP_STAGE
            // Moduł fast_exp stanowiący część logiki eml_cell
            fast_exp_ip exp_inst (.clk(clk),.in(x[i]),.out(exp_out[i]));
            always @(posedge clk) exp_reg[i] <= exp_out[i];
        end
    endgenerate

    // =========================================================================
    // KROK 2: Sumator turniejowy (Głębokość log2(N), dla N=8 wynosi 3)
    // =========================================================================
    reg [31:0] sum_l1 [0:3];
    reg [31:0] sum_l2 [0:1];
    reg [31:0] sum_total;

    generate
        for (i = 0; i < 4; i = i + 1) begin : ADD_L1
            wire [31:0] s1;
            float_add fadd_1 (.a(exp_reg[2*i]),.b(exp_reg[2*i+1]),.out(s1));
            always @(posedge clk) sum_l1[i] <= s1;
        end
        for (i = 0; i < 2; i = i + 1) begin : ADD_L2
            wire [31:0] s2;
            float_add fadd_2 (.a(sum_l1[2*i]),.b(sum_l1[2*i+1]),.out(s2));
            always @(posedge clk) sum_l2[i] <= s2;
        end
    endgenerate

    wire [31:0] s_final;
    // POPRAWKA D.4.4: Wejścia sum_l2[0] i sum_l2[1] zamiast dwukrotnie sum_l2
    float_add fadd_root (.a(sum_l2[0]),.b(sum_l2[1]),.out(s_final));
    
    always @(posedge clk) sum_total <= s_final;

    // =========================================================================
    // KROK 3: Obliczenie logarytmu naturalnego - fast_ln(S)
    // =========================================================================
    wire [31:0] ln_s_wire;
    reg  [31:0] ln_s_reg;
    
    // Analogiczna podselekcja logiki Minimax z komórki bazowej
    fast_ln_ip ln_inst (.clk(clk),.in(sum_total),.out(ln_s_wire));
    always @(posedge clk) ln_s_reg <= ln_s_wire;

    // =========================================================================
    // KROK 4: Propagacja sygnałów oryginalnych i ostateczne scalenie
    // =========================================================================
    // Moduł delay line synchronizuje dane o T = t_exp + t_tree + t_ln cykli
    wire [31:0] x_delayed [0:N-1];
    
    generate
        for (i = 0; i < N; i = i + 1) begin : DELAY_LINE
            delay_line_ip #(.WIDTH(32),.STAGES(10)) dline (
               .clk(clk),.in(x[i]),.out(x_delayed[i])
            );
            // x_i - ln(S) => dedykowany hardware Subtractor
            float_sub fsub_final (.a(x_delayed[i]),.b(ln_s_reg),.out(y[i]));
        end
    endgenerate

    // Symulacja rejestru przesuwnego dla sygnału validacji potoku
    reg [9:0] valid_delay;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) valid_delay <= 0;
        else valid_delay <= {valid_delay[8:0], valid_in};
    end
    
    assign valid_out = valid_delay[9];

endmodule
