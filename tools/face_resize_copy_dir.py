import argparse
import hashlib
import io
import os
from PIL import Image
import numpy as np
import onnxruntime as ort
import cv2


class Utils:
    def Md5(bytes):
        return hashlib.md5(bytes).hexdigest()

    def resize_by(image, percent):
        # 判断类型是否为PIL
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        width, height = image.size
        new_width = int(width * percent)
        new_height = int(height * percent)
        return image.resize((new_width, new_height), Image.LANCZOS)

    def resize_max(im, dst_w, dst_h):
        src_w, src_h = im.size

        if src_h > src_w:
            newWidth = dst_w
            newHeight = dst_w * src_h // src_w
        else:
            newWidth = dst_h * src_w // src_h
            newHeight = dst_h

        newHeight = newHeight // 8 * 8
        newWidth = newWidth // 8 * 8

        return im.resize((newWidth, newHeight), Image.Resampling.LANCZOS)

    def resize_min(im, dst_w, dst_h):
        src_w, src_h = im.size

        if src_h < src_w:
            newWidth = dst_w
            newHeight = dst_w * src_h // src_w
        else:
            newWidth = dst_h * src_w // src_h
            newHeight = dst_h

        newHeight = newHeight // 8 * 8
        newWidth = newWidth // 8 * 8

        return im.resize((newWidth, newHeight), Image.Resampling.LANCZOS)

    # 根据人脸进行剪裁
    def resize_crop(im, dst_w, dst_h, focus_point):

        src_w, src_h = im.size

        if src_h > src_w:
            newWidth = dst_w
            newHeight = dst_w * src_h // src_w
        else:
            newWidth = dst_h * src_w // src_h
            newHeight = dst_h

        newHeight = newHeight // 8 * 8
        newWidth = newWidth // 8 * 8

        im = im.resize((newWidth, newHeight), Image.Resampling.LANCZOS)

        
        left = (newWidth - dst_w) // 2
        top = (newHeight - dst_h) // 2

        right = left + dst_w
        bottom = top + dst_h


        focus_point_x, focus_point_y = focus_point 
        offset_w = dst_w // 8
        offset_h = dst_h // 8
        if focus_point_x < left + dst_w // 8:
            left = 0
            right = dst_w
        elif focus_point_x > right - dst_w // 8:
            right = newWidth
            left = newWidth - dst_w
        if focus_point_y < top + dst_h // 8:
            top = 0
            bottom = dst_h
        elif focus_point_y > bottom - dst_h // 8:
            bottom = newHeight
            top = newHeight - dst_h

        return im.crop((left, top, right, bottom))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, default="")
    parser.add_argument("-o", "--output_dir", type=str, default="")
    parser.add_argument("-m", "--face_model_path", type=str, default="")
    parser.add_argument("-s", "--score", type=float, default=0.8)
    args = parser.parse_args()
    limitscore = args.score
    sess = ort.InferenceSession(args.face_model_path)
    sess.set_providers(['CUDAExecutionProvider'])
    input_dir = args.input_dir
    output_dir = args.output_dir

    total = 0
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if not file.endswith(".png") and not file.endswith(".jpg"):
                continue
            total += 1
    
    import tqdm

    with tqdm.tqdm(total=total) as pbar:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if not file.endswith(".png") and not file.endswith(".jpg"):
                    continue
                
                file_path = os.path.join(root, file)
                pbar.update(1)
                pbar.set_description(f"Processing {file_path}")

                with open(file_path, "rb") as f:
                    md5_str = Utils.Md5(f.read())
                if os.path.exists(os.path.join(output_dir, md5_str + ".png")):
                    continue

                image = Image.open(file_path)
                image = Utils.resize_min(image, 1280, 1280)

                cv2_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                scores, bboxes, keypoints, aligned_imgs, landmarks, affine_matrices = sess.run(
                    None, {"input": cv2_img})
                if len(bboxes) != 1:
                    continue

                bboxes = bboxes[0]
                scores = scores[0]
                if limitscore > scores:
                    continue

                affine_matrices = affine_matrices[0]
                angle = np.degrees(np.arctan2(
                    affine_matrices[1][0], affine_matrices[0][0]))

                real_angle = 0
                if angle < -45 and angle > -135:
                    # 向左旋转90度
                    real_angle = 90
                elif angle > 45 and angle < 135:
                    # 向右旋转90度
                    real_angle = -90
                elif angle < -135 or angle > 135:
                    # 向左旋转180度
                    real_angle = 180

                if real_angle != 0:
                    continue

                # print(f"{file_path} bboxes: \n{bboxes} {scores} {real_angle}")
                

                center_point = (bboxes[0] + bboxes[2]) / \
                    2, (bboxes[1] + bboxes[3]) / 2
                image = Utils.resize_crop(image, 960, 1280, center_point)
                cv2_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                scores, bboxes, keypoints, aligned_imgs, landmarks, affine_matrices = sess.run(
                    None, {"input": cv2_img})
                if len(bboxes) != 1:
                    continue
                # cv2.imshow("image", cv2.cvtColor(
                #     np.array(image), cv2.COLOR_RGB2BGR))
                # cv2.waitKey(0) 
                
                output_path = os.path.join(output_dir, md5_str + ".png")
                # print(f"Save {file_path} to {output_path}")

                image.save(output_path)


if __name__ == "__main__":
    main()
