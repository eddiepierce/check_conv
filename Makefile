main:
	nvcc -lcuda main.c
cudnn:
	nvcc -lcuda -lcudnn cudnn.c
