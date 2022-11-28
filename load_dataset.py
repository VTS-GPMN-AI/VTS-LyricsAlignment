import glob
import json
def load_zalo_dataset():
  root_dataset_path = "/content/drive/MyDrive/Zalo_AI/dataset/"
  label_files = root_dataset_path + "train/labels"
  
  all_labels_path = glob.glob(label_files + "/*.json")
  result = []
  for file_name in all_labels_path[:2]:
    record = []
    with open(file_name) as f:
      content = json.load(f)
      # print(content)
      # id record
      idx = file_name.split("/")[-1].replace(".json", "")
      wav_file = "/content/drive/MyDrive/Zalo_AI/dataset/train/songs/" + idx + ".wav"
      se_record = []
      for sent_slice in content:
        se_record.append([[sent_slice['s'], sent_slice['e']], sent_slice['l']])
      record.extend((idx, wav_file, se_record))
      # print("record", record)
    result.append(record)

  return result
# res = load_zalo_dataset()
# print(len(res))