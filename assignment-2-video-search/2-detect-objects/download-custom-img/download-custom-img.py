from simple_image_download import simple_image_download as sid

response = sid.simple_image_download
# keywords = ["buildings", "bridge", "lakes", "ocean", "sewage", "leaf", "tree"]
keywords = ["plant grass", "flower"]
for kw in keywords:
    response().download(kw, limit=200)