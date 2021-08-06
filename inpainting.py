import cv2
import numpy as np
import tensorflow as tf
import glob 
import argparse
import os

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


INPUT_SIZE = 512  # input image size for Generator
ATTENTION_SIZE = 32 # size of contextual attention


def sort(str_lst):
    return [s for s in sorted(str_lst)]

# reconstruct residual from patches
def reconstruct_residual_from_patches(residual, multiple):
    residual = np.reshape(residual, [ATTENTION_SIZE, ATTENTION_SIZE, multiple, multiple, 3])
    residual = np.transpose(residual, [0,2,1,3,4])
    return np.reshape(residual, [ATTENTION_SIZE * multiple, ATTENTION_SIZE * multiple, 3])

# extract image patches
def extract_image_patches(img, multiple):
    h, w, c = img.shape
    img = np.reshape(img, [h//multiple, multiple, w//multiple, multiple, c])
    img = np.transpose(img, [0,2,1,3,4])
    return img

# residual aggregation module
def residual_aggregate(residual, attention, multiple):
    residual = extract_image_patches(residual, multiple * INPUT_SIZE//ATTENTION_SIZE)
    residual = np.reshape(residual, [1, residual.shape[0] * residual.shape[1], -1])
    residual = np.matmul(attention, residual)
    residual = reconstruct_residual_from_patches(residual, multiple * INPUT_SIZE//ATTENTION_SIZE)
    return residual

# resize image by averaging neighbors
def resize_ave(img, multiple):
    img = img.astype(np.float32)
    img_patches = extract_image_patches(img, multiple)
    img = np.mean(img_patches, axis=(2,3))
    return img

# pre-processing module
def pre_process(raw_img, raw_mask, multiple):

    raw_mask = raw_mask.astype(np.float32) / 255.
    raw_img = raw_img.astype(np.float32)

    # resize raw image & mask to desinated size
    large_img = cv2.resize(raw_img,  (multiple * INPUT_SIZE, multiple * INPUT_SIZE), interpolation = cv2. INTER_LINEAR)
    large_mask = cv2.resize(raw_mask, (multiple * INPUT_SIZE, multiple * INPUT_SIZE), interpolation = cv2.INTER_NEAREST)

    # down-sample large image & mask to 512x512
    small_img = resize_ave(large_img, multiple)
    small_mask = cv2.resize(raw_mask, (INPUT_SIZE, INPUT_SIZE), interpolation = cv2.INTER_NEAREST)

    # set hole region to 1. and backgroun to 0.
    small_mask = 1. - small_mask
    return large_img, large_mask, small_img, small_mask


# post-processing module
def post_process(raw_img, large_img, large_mask, res_512, img_512, mask_512, attention, multiple):

    # compute the raw residual map
    h, w, c = raw_img.shape
    low_base = cv2.resize(res_512.astype(np.float32), (INPUT_SIZE * multiple, INPUT_SIZE * multiple), interpolation = cv2.INTER_LINEAR)
    low_large = cv2.resize(img_512.astype(np.float32), (INPUT_SIZE * multiple, INPUT_SIZE * multiple), interpolation = cv2.INTER_LINEAR)
    residual = (large_img - low_large) * large_mask

    # reconstruct residual map using residual aggregation module
    residual = residual_aggregate(residual, attention, multiple)

    # compute large inpainted result
    res_large = low_base + residual
    res_large = np.clip(res_large, 0., 255.)

    # resize large inpainted result to raw size
    res_raw = cv2.resize(res_large, (w, h), interpolation = cv2.INTER_LINEAR)

    # paste the hole region to the original raw image
    mask = cv2.resize(mask_512.astype(np.float32), (w, h), interpolation = cv2.INTER_LINEAR)
    mask = np.expand_dims(mask, axis=2)
    res_raw = res_raw * mask + raw_img * (1. - mask)

    return res_raw.astype(np.uint8)

def wrap_frozen_graph(graph_def, inputs, outputs):
  def _imports_graph_def():
    tf.compat.v1.import_graph_def(graph_def, name="")
  wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
  import_graph = wrapped_import.graph
  return wrapped_import.prune(
      tf.nest.map_structure(import_graph.as_graph_element, inputs),
      tf.nest.map_structure(import_graph.as_graph_element, outputs))

def tf1_tf2(model_path):
  # path = "/content/sample-imageinpainting-HiFill/GPU_CPU/pb/hifill.pb"
  graph_def = tf.compat.v1.GraphDef()
  loaded = graph_def.ParseFromString(open(model_path,'rb').read())
  inception_func = wrap_frozen_graph(
    graph_def, inputs=['img:0', 'mask:0'],
    outputs=['inpainted:0', 'attention:0', 'mask_processed:0'])
  return inception_func

def inpaint(raw_img, 
            raw_mask, 
            multiple,
            model_path):

    # pre-processing
    
    img_large, mask_large, img_512, mask_512 = pre_process(raw_img, raw_mask, multiple)

    
    img_large = tf.convert_to_tensor(img_large)
    mask_large = tf.convert_to_tensor(mask_large)
    img_512 = tf.convert_to_tensor(img_512)
    mask_512 = tf.convert_to_tensor(mask_512[:,:,0:1])

    print(img_large.shape, img_large.dtype, type(img_large))
    print(mask_large.shape, mask_large.dtype, type(mask_large))
    print(img_512.shape, img_512.dtype, type(img_512))
    print(mask_512.shape, mask_512.dtype, type(mask_512))
    
    
    # print(f"some mask shape {mask_512[:,:,0:1].shape}")

    img_512 = img_512[tf.newaxis, ...]
    mask_512 = mask_512[tf.newaxis, ...]

    print(f"input shape {img_512.shape}")
    print(f"mask_512 {mask_512.shape}")

    # neural network
    HiFill_model = tf1_tf2(model_path)
    # print(mask_512.shape)
    inpainted_512, attention, mask_512 = HiFill_model(img_512, mask_512)

    inpainted_512 = np.array(inpainted_512)
    attention = np.array(attention)
    mask_512 = np.array(mask_512)
    img_large = np.array(img_large)
    mask_large = np.array(mask_large)
    
    img_512 = np.array(img_512[0])

    print(f"INPAINTED_512 {inpainted_512.shape} {inpainted_512.dtype} {type(inpainted_512)}")
    print(f"ATTENTION {attention.shape} {attention.dtype} {type(attention)}")
    print(f"MASK_512 {mask_512.shape} {mask_512.dtype} {type(mask_512)}")
    

    # post-processing
    res_raw_size = post_process(raw_img, img_large, mask_large, \
                 inpainted_512[0], img_512, mask_512[0], attention[0], multiple)

    return res_raw_size



def read_imgs_masks(images, marks):
    paths_img = glob.glob(images + '/*.*[gG]')
    paths_mask = glob.glob(marks + '/*.*[gG]')
    paths_img = sort(paths_img)
    paths_mask = sort(paths_mask)
    print('#imgs: ' + str(len(paths_img)))
    print('#imgs: ' + str(len(paths_mask)))
    print(paths_img)
    print(paths_mask)
    return paths_img, paths_mask




ap = argparse.ArgumentParser()

# args.images = '../samples/testset' # input image directory
# args.masks = '../samples/maskset' # input mask director
# args.output_dir = 'results' # output directory

ap.add_argument("-i", "--images", type=str, required=True,
                help="path input image on which we'll perform inpainting")
ap.add_argument("-m", "--masks", type=str, required=True,
                help="path input mask which corresponds to damaged areas")
ap.add_argument("-M", "--model", type=str, required=True,
                help="pretrained model path")                
ap.add_argument("-r", "--results", type=str, required=True,
                help="path the inpainted img dir path")
args = vars(ap.parse_args())

multiple = 6 # multiples of image resizing 

paths_img, paths_mask = read_imgs_masks(args["images"], args["masks"])

if not os.path.exists(args["results"]):
    os.makedirs(args["results"])

for path_img, path_mask in zip(paths_img, paths_mask):
  raw_img = cv2.imread(path_img)
  raw_mask = cv2.imread(path_mask)
  inpainted = inpaint(raw_img, raw_mask, multiple, args["model"])
  filename = args["results"] + '/' + os.path.basename(path_img)
  cv2.imwrite(filename + '_inpainted.jpg', inpainted)


