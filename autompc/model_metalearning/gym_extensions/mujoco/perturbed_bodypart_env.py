import tempfile
import xml.etree.ElementTree as ET
from gym import utils
from gym.envs.mujoco import MuJocoPyEnv

def ModifiedSizeEnvFactory(class_type):
    """class_type should be an OpenAI gym type"""

    class ModifiedSizeEnv(class_type, utils.EzPickle):

        def __init__(
                self,
                model_path,
                body_parts=["torso"],
                size_scale=1.0,
                *args,
                **kwargs):

            assert isinstance(self, MuJocoPyEnv)

            # find the body_part we want
            tree = ET.parse(model_path)
            for body_part in body_parts:
                # grab the geoms
                geom = tree.find(".//geom[@name='%s']" % body_part)
                sizes  = [float(x) for x in geom.attrib["size"].split(" ")]

                # rescale
                geom.attrib["size"] = " ".join([str(x * size_scale) for x in sizes ])  # the first one should always be the thing we want.

            # create new xml
            _, file_path = tempfile.mkstemp(text=True)
            file_path = file_path + ".xml"
            tree.write(file_path)

            # load the modified xml
            class_type.__init__(self, model_path=file_path)
            utils.EzPickle.__init__(self)

    return ModifiedSizeEnv
