from PIL import Image, ImageDraw

# Đọc hình ảnh từ file
input_image_path = "code/data/1Thẳng+6nghiêng nhẹ/trái 15.jpg"  # Thay đổi đường dẫn này thành đường dẫn đến hình ảnh của bạn
output_image_path = "output_image_with_points.png"

# Mở hình ảnh
image = Image.open(input_image_path)

# Tọa độ của 5 điểm
points = [(131, 241), (336, 285), (212, 375), (118, 460), (258, 489)]

# Tạo đối tượng vẽ
draw = ImageDraw.Draw(image)

# Vẽ các điểm lên hình ảnh
for point in points:
    radius = 5  # Bán kính của điểm
    draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius), fill='blue', outline='blue')

def Cut_face(model, device, img):
    img_size = 640
    conf_thres = 0.6
    iou_thres = 0.5

    im = np.array(img)
    img0 = copy.deepcopy(im)
    h0, w0 = im.shape[:2]
    r = img_size / max(h0, w0)
    if r != 1:
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())

    img = letterbox(img0, new_shape=imgsz)[0]
    img = img.transpose(2, 0, 1).copy()
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img)[0]
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)

    for _, det in enumerate(pred):
        im0 = im
        if len(pred[0]) == 1:
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], im0.shape).round()

                for j in range(det.size()[0]):
                    xyxy = det[j, :4].view(-1).tolist()
                    conf = det[j, 4].cpu().numpy()
                    landmarks = det[j, 5:15].view(-1).tolist()
                    class_num = det[j, 15].cpu().numpy()

                    im0 = crop_face(im0, xyxy, conf, landmarks, class_num)

    return len(pred[0]), im0
# Lưu hình ảnh mới với các điểm đã vẽ
image.save(output_image_path)

# Hiển thị thông báo đường dẫn file đã lưu
print(f"Hình ảnh mới đã được lưu tại: {output_image_path}")
