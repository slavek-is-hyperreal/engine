// -----------------------------------------------------------------------------
// Plik: rtl/eml_cell.v
// Opis: Sprzętowa komórka bazowa dla operatora EML: exp(x) - ln(y).
// Oparta na 4-stopniowym potokowaniu z zastosowaniem Minimax N=2 i N=3.
// POPRAWKA: Dodano korektę floor dla x < 0 (Bug D.4.1).
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module eml_cell #(
    parameter PIPELINE_STAGES = 4
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_in,
    input  wire [31:0] x,          // Format IEEE 754 float32
    input  wire [31:0] y,          // Format IEEE 754 float32 (y > 0)
    output reg  [31:0] result,     // Wynik ewaluacji eml(x, y)
    output reg         valid_out
);

    // Reprezentacja IEEE 754 stałych wielomianowych dla pełnej syntezowalności
    localparam [31:0] CONST_LOG2_E = 32'h3FB8AA3B; 
    localparam [31:0] CONST_EXP_A0 = 32'h3F80511A; 
    localparam [31:0] CONST_EXP_A1 = 32'h3F26AADD;
    localparam [31:0] CONST_EXP_A2 = 32'h3EB020C1; 
    localparam [31:0] CONST_LN_2   = 32'h3F317218; 
    localparam [31:0] CONST_LN_C1  = 32'h3F7CD59B; 
    localparam [31:0] CONST_LN_C2  = 32'hBED17DAF; 
    localparam [31:0] CONST_LN_C3  = 32'h3DEBCC19; 
    localparam [31:0] CONST_87     = 32'h42AE0000;
    localparam [31:0] CONST_MIN_87 = 32'hC2AE0000;
    localparam [31:0] CONST_1_0    = 32'h3F800000;

    // Rejestry opóźniające linię kontrolną dla sygnału valid_in
    reg v_st1, v_st2, v_st3;

    // =========================================================================
    // ETAP 1: Ograniczenie (Clamp), Iloczyn (x*LOG2_E), Ekstrakcja (y)
    // =========================================================================
    reg [31:0] w_st1;
    reg signed [31:0] e_int_st1;
    reg [31:0] m_float_st1;

    // Kombinacyjny clamp zapobiegający przepełnieniom
    wire [31:0] x_clamped = ($signed(x) > $signed(CONST_87))? CONST_87 : 
                            ($signed(x) < $signed(CONST_MIN_87))? CONST_MIN_87 : x;

    wire [31:0] fmul_w_out;
    float_mul inst_fmul_w (.a(x_clamped),.b(CONST_LOG2_E),.out(fmul_w_out));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v_st1 <= 1'b0;
            w_st1 <= 32'd0;
            e_int_st1 <= 32'd0;
            m_float_st1 <= 32'd0;
        end else begin
            v_st1 <= valid_in;
            w_st1 <= fmul_w_out;
            
            // Dekompozycja formatu IEEE 754 dla argumentu y 
            // Wykładnik (cecha) z odjęciem biasu (127)
            e_int_st1 <= $signed({1'b0, y[30:23]}) - 127;
            // Odbudowanie mantysy do pełnego float (zakres 1.0 - 2.0)
            m_float_st1 <= {1'b0, 8'd127, y[22:0]};
        end
    end

    // =========================================================================
    // ETAP 2: Rozdzielenie Całkowite/Ułamkowe, Subtrakcje dla f oraz u
    // =========================================================================
    reg [31:0] f_st2, u_st2;
    reg [31:0] scale_st2, e_float_st2;

    wire signed [31:0] w_int_raw;
    wire [31:0] i_float_wire;
    wire [31:0] e_float_wire;
    wire [31:0] fsub_f_out, fsub_u_out;

    // Przekształcenia typograficzne
    float_to_int inst_ftoi_w (.a(w_st1),.out(w_int_raw));
    
    // POPRAWKA D.4.1: Korekta floor dla w < 0.
    // Jeśli w < 0 i ma część ułamkową (w != i_float), i = i - 1.
    // Uproszczona implementacja w Verilog:
    wire signed [31:0] w_int_corr = (w_st1[31] && (fsub_f_out != 0)) ? w_int_raw - 1 : w_int_raw;

    int_to_float inst_itof_i (.a(w_int_corr),.out(i_float_wire));
    int_to_float inst_itof_e (.a(e_int_st1),.out(e_float_wire));
    
    // Obliczenie reszt: f = w - i, u = m - 1.0
    float_sub inst_fsub_f (.a(w_st1),.b(i_float_wire),.out(fsub_f_out));
    float_sub inst_fsub_u (.a(m_float_st1),.b(CONST_1_0),.out(fsub_u_out));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v_st2 <= 1'b0;
            f_st2 <= 32'd0; u_st2 <= 32'd0;
            scale_st2 <= 32'd0; e_float_st2 <= 32'd0;
        end else begin
            v_st2 <= v_st1;
            f_st2 <= fsub_f_out;
            u_st2 <= fsub_u_out;
            e_float_st2 <= e_float_wire;
            
            // Rekonstrukcja wektora skalowania poprzez mapowanie offsetu
            // scale = bitcast(u32(i + 127) << 23)
            scale_st2 <= {1'b0, w_int_corr[7:0] + 8'd127, 23'd0}; 
        end
    end

    // =========================================================================
    // ETAP 3: Horner Krok 1 (FMA) i Skalowanie logarytmu
    // =========================================================================
    reg [31:0] p1_st3, poly1_st3, eln2_st3;
    reg [31:0] f_st3, u_st3, scale_st3;

    wire [31:0] fma_p1_out, fma_poly1_out, fmul_eln2_out;

    // p1 = f*A2 + A1
    float_fma inst_fma_p1 (.a(f_st2),.b(CONST_EXP_A2),.c(CONST_EXP_A1),.out(fma_p1_out));
    // poly1 = u*C3 + C2
    float_fma inst_fma_poly1 (.a(u_st2),.b(CONST_LN_C3),.c(CONST_LN_C2),.out(fma_poly1_out));
    // Komponent liniowy = e*LN_2
    float_mul inst_fmul_eln2 (.a(e_float_st2),.b(CONST_LN_2),.out(fmul_eln2_out));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v_st3 <= 1'b0;
            p1_st3 <= 32'd0; poly1_st3 <= 32'd0; eln2_st3 <= 32'd0;
            f_st3 <= 32'd0; u_st3 <= 32'd0; scale_st3 <= 32'd0;
        end else begin
            v_st3 <= v_st2;
            p1_st3 <= fma_p1_out;
            poly1_st3 <= fma_poly1_out;
            eln2_st3 <= fmul_eln2_out;
            f_st3 <= f_st2;
            u_st3 <= u_st2;
            scale_st3 <= scale_st2;
        end
    end

    // =========================================================================
    // ETAP 4: Fuzja operatorów: Horner Krok 2, Agregacja Wynikowa
    // =========================================================================
    wire [31:0] p_final, poly_final;
    wire [31:0] u_poly, fast_exp_out, fast_ln_out, final_eml_out;

    // p = f*p1 + A0
    float_fma inst_fma_p2 (.a(f_st3),.b(p1_st3),.c(CONST_EXP_A0),.out(p_final));
    // poly = u*poly1 + C1
    float_fma inst_fma_poly2 (.a(u_st3),.b(poly1_st3),.c(CONST_LN_C1),.out(fma_poly_out));
    
    // Złożenie i normalizacja fast_exp oraz fast_ln w bloku kombinacyjnym
    float_mul inst_fmul_scale (.a(p_final),.b(scale_st3),.out(fast_exp_out));
    float_mul inst_fmul_upoly (.a(u_st3),.b(poly_final),.out(u_poly));
    float_add inst_fadd_ln    (.a(eln2_st3),.b(u_poly),.out(fast_ln_out));
    
    // Ostateczna ewaluacja operatora EML
    float_sub inst_fsub_final (.a(fast_exp_out),.b(fast_ln_out),.out(final_eml_out));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            result <= 32'd0;
        end else begin
            valid_out <= v_st3;
            result <= final_eml_out;
        end
    end

endmodule
