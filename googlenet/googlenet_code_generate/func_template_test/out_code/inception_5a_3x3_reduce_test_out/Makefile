cpp:
	g++ -DCPPSIM allocate_config.h template_config.h header.h nnet_relu.h nnet_bias.h nnet_common.h  nnet_mac.h nnet_conv_output_reuse.h nnet_buffer.h googlenet.h googlenet_conv.cpp -o cppExe
	./cppExe

hls:
	vivado_hls hls.tcl -l log/hls.log
	$(shell  rm -rf ./rpt/hls/)
	$(shell  mkdir -p ./rpt/hls/)
	cp hls_project/solution/syn/report/*.rpt ./rpt/hls
	#$(shell  rm -rf ./verilog)
	#$(shell  mkdir ./verilog)
	#cp hls_project/solution/impl/verilog/*.v ./verilog

#synth:
#	vivado -mode batch -source synth.tcl -log log/synth.log

#impl:
#	vivado -mode batch -source impl.tcl -log log/impl.log

#all:
#	vivado_hls hls.tcl -l log/hls.log
#	cp hls_project/solution/syn/report/*.rpt ./rpt/hls
#	vivado -mode batch -source synth.tcl -log log/synth.log
#	vivado -mode batch -source impl.tcl -log log/impl.log
