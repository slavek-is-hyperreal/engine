// -----------------------------------------------------------------------------
// Plik: tb/tb_eml_cell.v
// Opis: Weryfikacja dokładności komórki aproksymującej operator EML.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module tb_eml_cell();
    reg clk;
    reg rst_n;
    reg valid_in;
    reg [31:0] x_test;
    reg [31:0] y_test;
    wire [31:0] result;
    wire valid_out;

    eml_cell uut (
       .clk(clk),
       .rst_n(rst_n),
       .valid_in(valid_in),
       .x(x_test),
       .y(y_test),
       .result(result),
       .valid_out(valid_out)
    );

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    real res_real, expected_real, error_diff;
    
    initial begin
        rst_n = 0;
        valid_in = 0;
        x_test = 32'd0;
        y_test = 32'd0;
        #20 rst_n = 1;

        // ---------------------------------------------------------
        // TEST 1: eml(2.0, 1.0) -> exp(2.0) - ln(1.0) ≈ 7.389
        // ---------------------------------------------------------
        @(posedge clk);
        valid_in = 1;
        x_test = 32'h40000000;
        y_test = 32'h3F800000;
        
        @(posedge clk);
        // ---------------------------------------------------------
        // TEST 2: eml(0.0, exp(1.0)) -> exp(0.0) - ln(e) = 0.0
        // ---------------------------------------------------------
        x_test = 32'h00000000;
        y_test = 32'h402DF854;

        @(posedge clk);
        valid_in = 0;

        // Czekanie na zakończenie potoku
        wait(valid_out == 1);
        
        // Rozwiązanie TEST 1
        res_real = $bitstoshortreal(result);
        expected_real = 7.389;
        error_diff = res_real - expected_real;
        if (error_diff < 0) error_diff = -error_diff;
        
        $display("[%t] TEST 1. Wynik: %f, Oczekiwano: ~%f (Blad: %f)", $time, res_real, expected_real, error_diff);
            
        // Rozwiązanie TEST 2
        @(posedge clk);
        res_real = $bitstoshortreal(result);
        expected_real = 0.0;
        error_diff = res_real - expected_real;
        if (error_diff < 0) error_diff = -error_diff;
        
        $display("[%t] TEST 2. Wynik: %f, Oczekiwano: ~%f (Blad: %f)", $time, res_real, expected_real, error_diff);
            
        #50 $finish;
    end
endmodule
