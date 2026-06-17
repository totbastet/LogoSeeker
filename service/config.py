# Классификатор DINOv2
CLASSIFIER_PATH = r"./models/dinov2_arcface_background_epoch40.pth"

# Параметры классификатора DIBNOv2
CLASSIFIER_PIC_SIZE_MAX = 224
CLASSIFIER_BACKBONE_NAME = "vit_small_patch14_dinov2.lvd142m"
CLASSIFIER_EMBEDDING_SIZE = 384

# ====Параметры LoRA
CLASSIFIER_LORA_R = 16
CLASSIFIER_LORA_ALPHA = 16
CLASSIFIER_LORA_DROPOUT = 0.1
CLASSIFIER_LORA_TARGET_MODULES = ["attn.qkv"]

# ====Параметры проекционной головы
CLASSIFIER_PROJECTION_DIM_1 = 512
CLASSIFIER_PROJECTION_DROPOUT = 0.1


# Детектор Yolo
DETECTOR_PATH = r"./models/best.pt"

# ====Параметры детектора
DETECTOR_MODEL = "yolov8n.pt"
DETECTOR_THRESHOLD = 0.1
DETECTION_SIZE = 640


# Базы эмбеддингов
RESTRICTED_EMB_BASE_PATH = r"./bases/restricted_base.npy"
EMPLOYERS_EMB_BASE_PATH = r"./bases/employers_base.npy"
COMPETITORS_EMB_BASE_PATH = r"./bases/competitors_base.npy"
RESTRICTED_EMB_MAP_PATH = r"./bases/restricted_emb_map.json"
EMPLOYERS_EMB_MAP_PATH = r"./bases/employers_emb_map.json"
COMPETITORS_EMB_MAP_PATH = r"./bases/competitors_emb_map.json"

# Пороги принятия решения о совпадении эмбеддингов
RESTRICTED_SIMILARITY_THRESHOLD = 0.7
EMPLOYERS_SIMILARITY_THRESHOLD = 0.7
COMPETITORS_SIMILARITY_THRESHOLD = 0.7

# Трансформации
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

# for_tests
CLASSIFIER_TEST_IMAGE_PATH = r"./img/test.jpg"
DETECTOR_TEST_IMAGE_PATH = r"./img/test_2.jpg"
