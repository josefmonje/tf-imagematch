# Instructions...

Prepare model:
- Install bazel (check tensorflow's github for more info)
- Prepare folders and files for retraining
- Clone tensorflow
- Go to root of tensorflow
- bazel build tensorflow/examples/image_retraining:retrain
- bazel-bin/tensorflow/examples/image_retraining/retrain --image_dir /path/to/root_folder_name  --output_graph /path/output_graph.pb -- output_labels /path/output_labels.txt -- bottleneck_dir /path/bottleneck

** Training done. **

Outputs:
output_graph.pb
output_labels.txt

Use those outputs in this code as SOURCE_GRAPH and SOURCE_LABELS.

## Usage 

### Uses optional keyword arguments when provided

`im = ImageMatch(limit=10, source_graph='output_graph.pb', source_labels='output_labels.txt')`

### Uses local_settings when instantiated without arguments 

`from local_settings import SOURCE_GRAPH, SOURCE_LABELS`

### Uses default training output directory as default /tmp/

`im = ImageMatch()`

## Match an Image

`im.match('test.jpg')`


#### Labels and test image for flower retraining tutorial included

* test.jpg
* output_labels.txt

#### Graph is missing

* output_graph.pb 