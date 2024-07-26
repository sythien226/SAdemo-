from rembg import remove
from PIL import Image


input = Image.open('dog.png')

output = remove(input)

output.save('out.png')

# # output_image = remove(input_image, post_process_mask=True)
# img_io = BytesIO()
# # output_image.save(img_io, 'PNG')
# input_image.save(img_io, 'PNG')
# img_io.seek(0)
# download_file_name = Classification_Pig(input_image) + ".png"
# # return send_file(img_io, mimetype='image/png')  # Change download in separatre browser tab
# return send_file(img_io, mimetype='image/png', as_attachment=True, download_name=download_file_name)