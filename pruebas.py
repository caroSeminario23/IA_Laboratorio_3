import os

data_dir = 'bts_images_procesadas'

# Filtrar archivos que no sean directorios y excluir .DS_Store
class_names = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name)) and not name.startswith('.')]
print(f"Classes found: {class_names}, Count: {len(class_names)}")
