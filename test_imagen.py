from dotenv import load_dotenv
load_dotenv()

import json
from image_detector import analyze_image

print("=== FOTO REAL ===")
print(json.dumps(analyze_image(r"C:\Users\Romina\Desktop\Desktop\500817406_10233516187718445_5042738172609192388_n.jpg"), indent=2, ensure_ascii=False))

print("=== IMAGEN IA ===")
print(json.dumps(analyze_image(r"C:\Users\Romina\Desktop\Desktop\134238672039813558.jpg"), indent=2, ensure_ascii=False))

print("=== FOTO HDR ===")
print(json.dumps(analyze_image(r"C:\Users\Romina\Downloads\IMG_20240811_140124782_HDR.jpg"), indent=2, ensure_ascii=False))

print("=== FOTO HDR ===")
print(json.dumps(analyze_image(r"C:\Users\Romina\Downloads\IMG_20240816_100659817_HDR~2.jpg"), indent=2, ensure_ascii=False))


