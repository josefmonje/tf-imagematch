"""
Instructions...

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
"""
import os

import numpy as np
import tensorflow as tf

try:
    from local_settings import SOURCE_GRAPH, SOURCE_LABELS
except ImportError:
    # try bazel default output location on error
    SOURCE_GRAPH = '/tmp/output_graph.pb'
    SOURCE_LABELS = '/tmp/output_labels.txt'


class NullGraphException(Exception):
    pass


class NullLabelsException(Exception):
    pass


class NullImageException(Exception):
    pass


class ImageMatch(object):
    """
    Image Classifier object. Requires <source_graph>, <source_labels>. Optional <limit>.

    Accepts as keyword arguments upon instantiation, local_settings or hardcoded in this script.

    .match(image) prints top matches up to <limit> with scores, returns top match without score.
    Optionally limit top matches.
    """

    def __init__(self, **kwargs):
        """When instantiating, apply arguments to instance, check requirements, prepare graph."""
        for k, v in kwargs.items():
            setattr(self, k, v)

        self._check_requirements()
        self._create_graph()

    def _check_requirements(self):
        """Instantiate with requirements, import from local_settings or use default."""
        self.source_graph = getattr(self, 'source_graph', SOURCE_GRAPH)
        if not self.source_graph or not os.path.exists(self.source_graph):
            raise NullGraphException("The output graph (*.pb) is required.")

        self.source_labels = getattr(self, 'source_labels', SOURCE_LABELS)
        if not self.source_labels or not os.path.exists(self.source_labels):
            raise NullLabelsException("The labels file is required.")

    def _create_graph(self):
        """Create graph from saved GraphDef file."""
        with open(self.source_graph, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

    def match(self, target_image=None):
        """Predict matches and store results."""
        if not target_image:
            raise NullImageException("A target image is required.")
        elif not os.path.exists(target_image):
            raise NullImageException("Target image does not exist.")

        with tf.Session() as session, open(target_image, 'rb') as image:
            tensor = session.graph.get_tensor_by_name('final_result:0')  # retrained
            predictions = session.run(tensor, {'DecodeJpeg/contents:0': image.read()})  # noqa
            self.predictions = np.squeeze(predictions)
            self.results = self.predictions.argsort()[::-1]  # results in ascending order

            if self.limit:
                self.results = self.results[:self.limit]
        self.show_matches()

    def show_matches(self):
        """Print match scores, return match."""
        with open(self.source_labels, 'r') as labels:
            print('Matches:')
            labels = [line.replace("\n", "") for line in labels.readlines()]
            for result in self.results:
                label = labels[result]
                score = self.predictions[result]
                print('{} ({:.4f})'.format(label, score))

            return labels[self.results[0]]


if __name__ == '__main__':
    cls = ImageMatch(limit=10, source_graph='output_graph.pb', source_labels='output_labels.txt')  # noqa
    cls.match('test.jpg')
