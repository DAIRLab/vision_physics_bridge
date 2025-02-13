import paramiko
import numpy as np
from skimage.io import imread
from io import BytesIO
from shutil import copy2
import os

def create_ssh_client(server, port, user, password):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port=port, username=user, password=password)
    return client

def find_offset_from_long_to_short(long_folder, short_folder, offset=None):
    '''
    Find the offset between the short and long images. short[0] == long[0+offset]
    '''
    # List all images in the short folder
    short_images = sorted(os.listdir(short_folder))
    print(f'Number of short_images: {len(short_images)}')

    # List all images in the long folder
    long_images = sorted(os.listdir(long_folder))
    print(f'Number of long_images: {len(long_images)}')

    if offset is not None:
        return offset, len(long_images) - len(short_images)

    short_images_data = {img: imread(os.path.join(short_folder, img)) for img in short_images}
    long_images_data = {img: imread(os.path.join(long_folder, img)) for img in long_images}

    # Load long images and compare to find offset
    for offset in range(0, len(long_images) - len(short_images)+1):
        for i, short_img in enumerate(short_images):
            if i % 10 == 0:
                if not np.array_equal(long_images_data[long_images[i+offset]], short_images_data[short_img]):
                    break
            return offset, len(long_images) - len(short_images)
    return None, len(long_images) - len(short_images)
    

def find_offset_from_short_remote_to_long_local(ssh_client, remote_folder, local_folder):
    '''
    Find the offset between the local and remote images. long_local[0+offset] == short_remote[0]
    '''
    # List all images in the local folder
    print(f"ls {local_folder}")
    local_images = sorted(os.listdir(local_folder))
    local_images_data = {img: imread(os.path.join(local_folder, img)) for img in local_images} 
    # TODO: [..., ::-1] was used but actually not needed for robotocc data. When was it needed?
    print(f'Number of local_images: {len(local_images_data.keys())}')
    
    # local_images_data = {img: local_images_data[img][..., ::-1] for img in local_images_data}

    # List all images in the remote folder
    if ssh_client is not None:
        print(f"ls {remote_folder}")
        stdin, stdout, stderr = ssh_client.exec_command(f"ls {remote_folder}")
        remote_images = sorted(stdout.read().decode().split())
        print(f'Number of remote_images: {len(remote_images)}')
    else:
        print(f"ls {remote_folder}")
        remote_images = sorted(os.listdir(remote_folder))
        remote_image_data = {img: imread(os.path.join(remote_folder, img)) for img in remote_images}
        print(f'Number of remote_images: {len(remote_images)}')
    
    def plot_two_imgs(i, offset):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(local_images_data[local_images[i+offset]])
        ax[1].imshow(remote_image_data[remote_images[i]])
        plt.show()

    # Load remote images and compare to find offset
    for offset in range(len(local_images) - len(remote_images), -1, -1):
        synced = True
        for i, remote_img in enumerate(remote_images):
            if i % 10 == 0:
                if ssh_client is None:
                    if not np.array_equal(local_images_data[local_images[i+offset]], remote_image_data[remote_img]):
                        # print(f"Check failed at offset {offset}, local image {i+offset} and remote image {i}.")
                        synced = False
                        break
                else:
                    stdin, stdout, stderr = ssh_client.exec_command(f"cat {remote_folder}/{remote_img}")
                    remote_img_data = imread(BytesIO(stdout.read()))
                    if not np.array_equal(local_images_data[local_images[i+offset]], remote_img_data):
                        synced = False
                        break
        if synced:
            break
    if not synced:
        print("No offset matches the images. Check image content or adjust parameters.")
        breakpoint()
    return offset, synced

def copy_files_to_a_new_folder(folder, new_folder):
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    for img in os.listdir(folder):
        copy2(os.path.join(folder, img), os.path.join(new_folder, img))

def copy_and_rename_files_from_long_to_short(long_folder, short_folder, offset, diff_len):
    '''
    Copy files from long to short folder, renaming them with an offset.
    short[0] == long[0+offset]
    '''
    if not os.path.exists(short_folder):
        os.makedirs(short_folder)

    # sort the long images
    long_files = sorted(os.listdir(long_folder))
    num_files = len(long_files) - diff_len

    for long_img in long_files[offset:offset+num_files]:
        original_img_path = os.path.join(long_folder, long_img)
        img_data = imread(original_img_path)
        
        new_index = int(long_img.split('.')[0]) - offset
        new_img_path = os.path.join(short_folder, f"{str(new_index).zfill(4)}.png")
        
        copy2(original_img_path, new_img_path)

def copy_and_rename_files_from_short_remote_to_long_local(ssh_client, remote_folder, local_target_folder, offset):
    '''
    long_local[0+offset] == short_remote[0]
    '''
    if ssh_client is not None:
        stdin, stdout, stderr = ssh_client.exec_command(f"ls {remote_folder}")
        remote_images = sorted(stdout.read().decode().split())
    else:
        remote_images = sorted(os.listdir(remote_folder))

    if not os.path.exists(local_target_folder):
        os.makedirs(local_target_folder)

    for remote_img in remote_images:
        original_mask_path = f"{remote_folder}/{remote_img}"
        if ssh_client is not None:
            stdin, stdout, stderr = ssh_client.exec_command(f"cat {original_mask_path}")
            mask_data = stdout.read()
        
            new_index = int(remote_img.split('.')[0]) + offset
            new_mask_path = f"{local_target_folder}/{str(new_index).zfill(4)}.png"
            
            with open(new_mask_path, 'wb') as f:
                f.write(mask_data)
        else:
            new_index = int(remote_img.split('.')[0]) + offset
            new_mask_path = f"{local_target_folder}/{str(new_index).zfill(4)}.png"
            copy2(original_mask_path, new_mask_path)

def sync_from_remote_and_local(ssh_client, remote_root_folder, remote_toss_id, local_root_folder, local_toss_id):
    local_folder = f"{local_root_folder}/{local_toss_id}/rgb"
    remote_folder = f"{remote_root_folder}/{remote_toss_id}/rgb"
    offset, synced = find_offset_from_short_remote_to_long_local(ssh_client, remote_folder, local_folder)

    print(f"Offset: {offset}, Synced: {synced}")
    if synced:
        local_annotations = f"{local_root_folder}/{local_toss_id}/Annotations"
        remote_annotations = f"{remote_root_folder}/{remote_toss_id}/Annotations"
        print(f"Copying annotations from {remote_annotations} to {local_annotations}.")
        copy_and_rename_files_from_short_remote_to_long_local(ssh_client, remote_annotations, local_annotations, offset)

        # if offset == 0 and synced:
        #     local_masks = f"{local_root_folder}/{local_toss_id}/masks"
        #     remote_masks = f"{remote_root_folder}/{remote_toss_id}/masks"
        #     print(f"Copying masks from {remote_masks} to {local_masks} as well.")
        #     copy_and_rename_files_from_short_remote_to_long_local(ssh_client, remote_masks, local_masks, offset)
    else:
        print("No valid offset found. No operation performed.")

def prop_mask_down(long_root_folder, long_toss_id, short_root_folder, short_toss_id, offset=None):
    long_folder = f"{long_root_folder}/{long_toss_id}/rgb"
    short_folder = f"{short_root_folder}/{short_toss_id}/rgb"
    offset, diff_len = find_offset_from_long_to_short(long_folder, short_folder, offset)

    print(f"Offset: {offset}, Diff_len: {diff_len}")
    if offset is not None:
        long_masks = f"{long_root_folder}/{long_toss_id}/masks"
        short_masks = f"{short_root_folder}/{short_toss_id}/masks"
        print(f"Copying masks from {long_masks} to {short_masks}")
        copy_and_rename_files_from_long_to_short(long_masks, short_masks, offset, diff_len)
    else:
        print("No suitable offset found. Check image content or adjust parameters.")

def prop_annotation_up(from_root_folder, from_loss_id, to_root_folder, to_loss_id):
    from_folder = f"{from_root_folder}/{from_loss_id}/Annotations"
    to_folder = f"{to_root_folder}/{to_loss_id}/Annotations"
    print(f"Copying annotations from {from_folder} to {to_folder}.")
    copy_files_to_a_new_folder(from_folder, to_folder)

# # Setup SSH connection
# server = '158.130.72.130'#
# port = 22
# user = 'cnets-vision'
# password = '' #type the password here
# ssh_client = create_ssh_client(server, port, user, password)

# Specify the local root folder and the remote root folder. Specify the local toss id and the remote toss id. Check the offset. 
# If the offset is zero and the number of files are the same, directly copy masks. 
# If the offset can be found, copy the annotations. 
# If the offset cannot be found (None), print a message.

remote_folder_b = '/home/cnets-vision/mengti_ws/structural_tests/bundlenets/data'
# remote_annotations = '/home/cnets-vision/mengti_ws/structural_tests/bundlenets/data/cube_2/Annotations'
local_folder_a = '/mnt/data0/minghz/repos/bundlenets/data'
# local_target_folder = '/mnt/data0/minghz/repos/bundlenets/data/cube_2/Annotations'

# cube, bottle, half, milk, toblerone, prism, egg, napkin, box
# bakingbox, burger, cardboard, chocolate, cream, croc, crushedcan, duck, gallon, greencan, hotdog, icetray, mug, oatly, pinkcan, stapler, styrofoam, toothpaste


# for obj_name in ['bakingbox', 'burger', 'cardboard', 'chocolate', 'cream', 'croc', 'crushedcan', 'duck', 'gallon', 'greencan', 'hotdog', 'icetray', 'mug', 'oatly', 'pinkcan', 'stapler', 'styrofoam', 'toothpaste']:
### all objects
# objs=("bakingbox" "burger" "cardboard" "chocolate" "cream" "croc" "crushedcan" "duck" "gallon" "greencan" "hotdog" "icetray" "mug" "oatly" "pinkcan" "stapler" "styrofoam" "toothpaste" "prism" "egg" "napkin" "cube" "bottle" "half" "milk" "toblerone")
### objects with gt mesh
# objs=("bakingbox" "cardboard" "croc" "crushedcan" "gallon" "greencan" "oatly" "pinkcan" "stapler" "styrofoam" "egg" "napkin" "cube" "bottle" "half" "milk")
### robot objects 
# objs=("robot_bakingbox" "robot_greencan" "robot_milk" "robot_oatly" "robot_stapler")
assets=['bakingbox_1-8', 'bakingbox_9-11', 'bakingbox_12-15', 'bottle_1-7', 'egg_1-6', 'milk_1-5', 'oatly_1-6', 'oatly_7-12', 'styrofoam_1-6', 'styrofoam_7-10', 'toblerone_1-5', 'toblerone_6-11']
# for obj_name in ['robot_greencan', 'robot_milk', 'robot_oatly', 'robot_stapler']:
for asset in assets:
    obj_name, target_ids = asset.split('_')
    obj_name = 'robotocc_'+obj_name
    target_id_start, target_id_end = target_ids.split('-')
    target_id_start, target_id_end = int(target_id_start), int(target_id_end)
    # sync_from_remote_and_local(ssh_client, remote_folder_b, obj_name+'_1', local_folder_a, obj_name+'_1')
    # sync_from_remote_and_local(ssh_client, remote_folder_b, obj_name+'_2', local_folder_a, obj_name+'_2')
    # sync_from_remote_and_local(ssh_client, remote_folder_b, obj_name+'_3', local_folder_a, obj_name+'_3')
    # sync_from_remote_and_local(ssh_client, remote_folder_b, obj_name+'_4', local_folder_a, obj_name+'_4')
    # sync_from_remote_and_local(ssh_client, remote_folder_b, obj_name+'_5', local_folder_a, obj_name+'_5')
    # sync_from_remote_and_local(ssh_client, remote_folder_b, obj_name+'_2', local_folder_a, obj_name+'_1-2')
    # sync_from_remote_and_local(ssh_client, remote_folder_b, obj_name+'_3', local_folder_a, obj_name+'_1-3')
    # sync_from_remote_and_local(ssh_client, remote_folder_b, obj_name+'_4', local_folder_a, obj_name+'_1-4')
    # sync_from_remote_and_local(ssh_client, remote_folder_b, obj_name+'_5', local_folder_a, obj_name+'_1-5')

    # sync_from_remote_and_local(None, local_folder_a, obj_name+'_2', local_folder_a, obj_name+'_1-2')
    # sync_from_remote_and_local(None, local_folder_a, obj_name+'_3', local_folder_a, obj_name+'_1-3')
    # sync_from_remote_and_local(None, local_folder_a, obj_name+'_4', local_folder_a, obj_name+'_1-4')
    # sync_from_remote_and_local(None, local_folder_a, obj_name+'_5', local_folder_a, obj_name+'_1-5')

    # prop_annotation_up(local_folder_a, obj_name+'_1', local_folder_a, obj_name+'_1-2')
    # prop_annotation_up(local_folder_a, obj_name+'_1-2', local_folder_a, obj_name+'_1-3')
    # prop_annotation_up(local_folder_a, obj_name+'_1-3', local_folder_a, obj_name+'_1-4')
    # prop_annotation_up(local_folder_a, obj_name+'_1-4', local_folder_a, obj_name+'_1-5')

    for target_id in range(target_id_start, target_id_end+1):
        sync_from_remote_and_local(None, local_folder_a, obj_name+'_'+str(target_id), local_folder_a, obj_name+'_'+str(target_ids))

    ### do annotation on the largest (not in this script)

    # ## do propogation of the masks from the largest to the smallest
    # prop_mask_down(local_folder_a, obj_name+'_1-5', local_folder_a, obj_name+'_1-4', offset=0)
    # prop_mask_down(local_folder_a, obj_name+'_1-4', local_folder_a, obj_name+'_1-3', offset=0)
    # prop_mask_down(local_folder_a, obj_name+'_1-3', local_folder_a, obj_name+'_1-2', offset=0)
    # prop_mask_down(local_folder_a, obj_name+'_1-2', local_folder_a, obj_name+'_1', offset=0)

    # prop_mask_down(local_folder_a, obj_name+'_1-5', local_folder_a, obj_name+'_5')
    # prop_mask_down(local_folder_a, obj_name+'_1-5', local_folder_a, obj_name+'_4')
    # prop_mask_down(local_folder_a, obj_name+'_1-5', local_folder_a, obj_name+'_3')
    # prop_mask_down(local_folder_a, obj_name+'_1-5', local_folder_a, obj_name+'_2')

    # prop_mask_down(local_folder_a, obj_name+'_1-2', local_folder_a, obj_name+'_2')