import zipfile
import os

from tqdm import tqdm


def extract_specific_subfolders(zip_path, target_path, base_folder, subfolders):
    """
    Extract specific subfolders from within a base folder in a ZIP archive.

    :param zip_path: Path to the ZIP file.
    :param target_path: Path where the subfolders should be extracted.
    :param base_folder: The base folder in the ZIP file from where to extract the subfolders.
    :param subfolders: List of subfolder names to extract.
    """
    for subfolder in subfolders:
        print(f'{base_folder}/{subfolder}/')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # List all the files and directories in the zip
        for member in tqdm(zip_ref.namelist()):
            # if 'CLS-LOC/train' in member:
            #     print(member)

            # Check if the member is in one of the subfolders we want, and is within the base folder
            if any(member.startswith(f"{base_folder}/{subfolder}/") for subfolder in subfolders):
                # Extract the file
                print(f"Extracting {member} to {target_path}")
                zip_ref.extract(member, target_path)

# Example usage
zip_file_path = '/home/student/data/imagenet-object-localization-challenge.zip'
destination_path = '/home/student/data/imagenet/'
subfolders_to_extract =  'n01558993 n01601694 n01669191 n01751748 n01755581 n01756291 n01770393 n01855672 n01871265 n02018207 n02037110 n02058221 n02087046 n02088632 n02093256 n02093754 n02094114 n02096177 n02097130 n02097298 n02099267 n02100877 n02104365 n02105855 n02106030 n02106166 n02107142 n02110341 n02114855 n02120079 n02120505 n02125311 n02128385 n02133161 n02277742 n02325366 n02364673 n02484975 n02489166 n02708093 n02747177 n02835271 n02906734 n02909870 n03085013 n03124170 n03127747 n03160309 n03255030 n03272010 n03291819 n03337140 n03450230 n03483316 n03498962 n03530642 n03623198 n03649909 n03710721 n03717622 n03733281 n03759954 n03775071 n03814639 n03837869 n03838899 n03854065 n03929855 n03930313 n03954731 n03956157 n03983396 n04004767 n04026417 n04065272 n04200800 n04209239 n04235860 n04311004 n04325704 n04336792 n04346328 n04380533 n04428191 n04443257 n04458633 n04483307 n04509417 n04515003 n04525305 n04554684 n04591157 n04592741 n04606251 n07583066 n07613480 n07693725 n07711569 n07753592 n11879895'


extract_specific_subfolders(zip_file_path, destination_path, 'ILSVRC/Data/CLS-LOC/train',subfolders_to_extract.split(' '))