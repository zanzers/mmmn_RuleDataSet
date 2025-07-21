from algorithm.domain import mmmn_domain
from algorithm.tools.functions import *
import cv2
import json
from pathlib import Path





def build_ruleSet(img_dir: str, output_path: str="features_data.json"):

    data = {}
    
    for label in ['0', '1', '2']:
        folder = Path(img_dir) / label
        if not folder.exists():
            continue

        for img_file in folder.glob("*.[jJpP][nNpP][gG]*"):
            image_name = img_file.name  
            print(f"\n[PROCESSING] {image_name} (Categoty={label}")

            try:
                features = run(str(img_file), label, image_name)
                features["label"] = int(label)
                data[image_name] = features
            
            except Exception as e:
                print(f"[ERROR] Failed to process {image_name}: {e}")
                continue

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"[DONE] Feature data saved to {output_path}")






def run(img):
    global global_loading
    global_loading = False

    t = threading.Thread(target=loader_anim, args=(f"Analyzing {Path(img).name}",))
    t.start()

    result = mmmn_domain(img)
    global_loading = True
    t.join()

    


if __name__ == "__main__":

    build_ruleSet('data/training_img')



