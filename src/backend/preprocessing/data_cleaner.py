import os
from PIL import Image
import face_recognition

def rename_and_clean_images(image_names: list):

    curr_dir = os.path.dirname(os.path.abspath(__file__))

    for img_name in image_names:

        raw_img_dir_rel_path = os.path.join('..', '..', '..', 'data', 'raw-images', f'{img_name}-raw-images-new')
        cleaned_img_dir_rel_path = os.path.join('..', '..', '..', 'data', 'cleaned-images', f'{img_name}-cleaned-images')

        raw_img_dir_abs_path = os.path.normpath(os.path.join(curr_dir, raw_img_dir_rel_path))
        cleaned_img_dir_abs_path = os.path.normpath(os.path.join(curr_dir, cleaned_img_dir_rel_path))

        if not os.path.isdir(cleaned_img_dir_abs_path):
            os.makedirs(cleaned_img_dir_abs_path)
            print(f'created directory: {cleaned_img_dir_abs_path}')

        if os.listdir(cleaned_img_dir_abs_path):
            continue

        #raw_image_dir = os.fsencode(raw_img_dir_abs_path)
        raw_image_dir = raw_img_dir_abs_path

        counter = 1
        for filename in os.listdir(raw_image_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                source_path = os.path.join(raw_image_dir, filename)

                with Image.open(source_path) as img:
                    # Convert to RGB for saving PNGs or non-RGB JPEGs to JPEG
                    rgb_img = img.convert('RGB')

                new_filename = f"{img_name}-cleaned-{counter}.jpg"
                destination_path = os.path.join(cleaned_img_dir_abs_path, new_filename)

                rgb_img.save(destination_path, "JPEG")
                counter += 1
                continue
            else:
                continue
        
        print(f'saved {len(cleaned_img_dir_abs_path)} JPG files from {len(os.listdir(raw_image_dir))} files into directory: {cleaned_img_dir_abs_path}')
    
    return


def crop_face(image_names: list):
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    for img_name in image_names:

        cleaned_img_dir_rel_path = os.path.join(curr_dir, '..', '..', '..', 'data', 'cleaned-images', f'{img_name}-cleaned-images')
        cropped_img_dir_rel_path = os.path.join(curr_dir, '..', '..', '..', 'data', 'cropped-images', f'{img_name}')

        cleaned_img_dir_rel_path = os.path.normpath(cleaned_img_dir_rel_path)
        cropped_img_dir_rel_path = os.path.normpath(cropped_img_dir_rel_path)

        if not os.path.isdir(cropped_img_dir_rel_path):
            os.makedirs(cropped_img_dir_rel_path)
            print(f'created directory: {cropped_img_dir_rel_path}')

        if os.listdir(cropped_img_dir_rel_path):
            continue

        counter = 1
        for filename in os.listdir(cleaned_img_dir_rel_path):  
            source_path = os.path.join(cleaned_img_dir_rel_path, filename)

            try:
                image = face_recognition.load_image_file(source_path)
                face_locations = face_recognition.face_locations(image)

                if len(face_locations) != 1:
                    continue  # Discard images with 0 or >1 faces

                top, right, bottom, left = face_locations[0]
                face_image = image[top:bottom, left:right]
                pil_image = Image.fromarray(face_image)

                new_filename = f"{img_name}-face-{counter}.jpg"
                destination_path = os.path.join(cropped_img_dir_rel_path, new_filename)
                pil_image.save(destination_path)
                counter += 1

            except Exception as e:
                print('error')
                continue

        print(f'saved cropped face images into: {cropped_img_dir_rel_path}')

    return