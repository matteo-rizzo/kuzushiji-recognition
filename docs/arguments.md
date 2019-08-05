# Arguments for the YOLO model

* --imgdir         path to testing directory with images
* --binary         path to .weights directory
* --config         path to .cfg directory
* --dataset        path to dataset directory
* --labels         path to labels file
* --backup         path to backup folder
* --summary        path to TensorBoard summaries directory
* --annotation     path to annotation directory
* --threshold      detection threshold
* --model          configuration of choice
* --trainer        training algorithm
* --momentum       applicable for rmsprop and momentum optimizers
* --verbalise      say out loud while building graph
* --train          train the whole net
* --load           how to initialize the net? Either from .weights or a checkpoint, or even from scratch
* --savepb         save net and weight to a .pb file
* --gpu            how much gpu (from 0.0 to 1.0)
* --gpuName        GPU device name
* --lr             learning rate
* --keep           Number of most recent training results to save
* --batch          batch size
* --epoch          number of epoch
* --save           save checkpoint every ? training examples
* --demo           demo on webcam
* --queue          process demo in batch
* --json           Outputs bounding box information in json format.
* --saveVideo      Records video from input video or camera
* --pbLoad         path to .pb protobuf file (metaLoad must also be specified)
* --metaLoad       path to .meta file generated during --savepb that corresponds to .pb file

