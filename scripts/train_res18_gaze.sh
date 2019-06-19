
GLOG_logtostderr=1 ../../../build/tools/caffe train \
  --solver=quicker_solver.prototxt \
  --gpu=0  2>&1 | tee U_res_log_p02c.txt
