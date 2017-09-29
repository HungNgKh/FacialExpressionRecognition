import os


PATH = '../saved/'

for root, dirs, files in os.walk(PATH):
     for file in files:
         os.remove(os.path.join(root, file))