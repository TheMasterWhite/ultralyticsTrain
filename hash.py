import hashlib
import os
import shutil
import concurrent.futures


# 计算文件的哈希值
def calculate_hash(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


# 删除文件夹中的重复文件，只保留一个
def delete_duplicates(directory):
    hashes = {}
    num = 0
    for filename in [i for i in os.listdir(directory)]:
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_hash = calculate_hash(file_path)
            if file_hash in [i for i in hashes]:
                # 如果哈希值已存在，删除当前文件
                os.remove(file_path)
                num += 1
                print(f"Deleted duplicate file: {file_path}")
            else:
                # 如果哈希值不存在，保存文件路径
                hashes[file_hash] = file_path
    print(f"Deleted {num} files")


if __name__ == "__main__":
    directory = ""
    delete_duplicates(directory)
