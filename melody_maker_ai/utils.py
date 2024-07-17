import hashlib
import uuid

def file_hash(filename):
    """파일의 내용을 기반으로 SHA-256 해시 값을 계산합니다."""
    sha256_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        # 파일을 청크 단위로 읽어 해시를 업데이트합니다.
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def generate_uuid():
    """파일에 대한 고유한 UUID를 생성합니다."""
    return str(uuid.uuid4())

def get_file_extension(filename):
    """파일의 확장자를 반환합니다."""
    return filename.split(".")[-1]