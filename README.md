# autoboundserver
Server for [AutoBound](https://github.com/BBloggsbott/autobound)<br/>

Currently implemented End points:

* `/` - Used to generate nodes for a building in the aerial image received in the request
* `/generateNodesTest` - Used by the unit test for the JOSM plugin
* `/dataCollector` - Used to collect data (original image, segmented image and other metadata) of buildings using the `Collect Data - AutoBound` oprtion under the `Tools` menu in JOSM.

## Using your own models
Save your the model you have built and copy them to a directory of your choice. Make sure to modify the path to your model [here](https://github.com/BBloggsbott/autoboundserver/blob/7c936973d69f78f4d1597446f16889f9e763dc07/autoboundserver/autoboundserver.py#L39). Update the model architecture [here](https://github.com/BBloggsbott/autoboundserver/blob/5e5d9e1a45b68cb4f92a0282ff40ddd7b99dc95f/autoboundserver/model_utils.py#L8) and import it in the necessary places.

Server for [AutoBound](https://github.com/BBloggsbott/autobound). Read more about it in the wiki [here](https://wiki.openstreetmap.org/wiki/JOSM/Plugins/AutoBound).

The docker image for this server can be found [here](https://cloud.docker.com/u/bbloggsbott/repository/docker/bbloggsbott/autoboundserver).

If you are interested in contributing, please read the contributing guidelines [here](https://github.com/BBloggsbott/autoboundserver/blob/master/CONTRIBUTING.md).