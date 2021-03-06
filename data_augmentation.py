import random
import copy
def data_augmentation(input_ids,input_mask,label_ids,seq_length,num_token):
    real_len = sum(input_mask)
    input_ids = copy.deepcopy(input_ids)
    input_mask = copy.deepcopy(input_mask)
    label_ids = copy.deepcopy(label_ids)
    if random.random() < 2/3:
        for i in range(5):
            insert_place, input_token = random.randint(1, real_len - 2), random.randint(0, num_token - 1)
            if random.random()<1:
                input_ids.insert(insert_place, input_token)
            else:
                input_ids.insert(insert_place, input_ids[insert_place])
            input_mask.insert(insert_place, 1)
            label_ids.insert(insert_place, 100)
    else:
        for i in range(seq_length-5,seq_length):
            insert_place = i
            input_ids.insert(insert_place, 0)
            input_mask.insert(insert_place, 1)
            label_ids.insert(insert_place, 100)
    input_ids = input_ids[:seq_length]
    input_mask = input_mask[:seq_length]
    label_ids = label_ids[:seq_length]
    return input_ids, input_mask, label_ids