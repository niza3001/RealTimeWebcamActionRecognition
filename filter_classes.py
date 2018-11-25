from dataloader.split_train_test_video import UCF101_splitter

data_handler = UCF101_splitter('/hdd/NLN/UCF_list/', None)
data_handler.get_action_index()
class_to_idx = data_handler.action_label
idx_to_class = {v: k for k, v in class_to_idx.iteritems()}


def write_train_set(fname, idx_to_class, out_file):
    with open(fname) as f:
        content = f.readlines()
        content = [x.strip('\r\n') for x in content if x.strip('\r\n').split(" ")[1] in idx_to_class.keys()]
        content = list(set(content))
        content.sort(key=lambda x: x.split('/')[0])
        with open(out_file, 'w') as f2:
            for line in content:
                f2.write("%s\n" % line)


def convert_train_set(fname, oldi_to_newi, out_file):
    with open(fname) as f:
        content = f.readlines()
        filename = [x.strip('\r\n').split(' ')[0] for x in content]
        old_labels = [x.strip('\r\n').split(' ')[1] for x in content]
        new_labels = [oldi_to_newi[old_label] for old_label in old_labels]
        with open(out_file, 'w') as f2:
            for filename, new_label in zip(filename, new_labels):
                f2.write("{} {}\n".format(filename, new_label))


def write_test_set(fname, class_to_idx, out_file):
    with open(fname) as f:
        content = f.readlines()
        content = [x.strip('\r\n') for x in content]
        content = list(set(content))
        content.sort(key=lambda x: x.split('/')[0])
        with open(out_file, 'w') as f2:
            for line in content:
                action = line.split('/')[0]
                if action in class_to_idx.keys():
                    f2.write("%s\n" % line)


def create_new_class_idx(fname, out_file):
    with open(fname) as f:
        content = f.readlines()
        classes = [x.strip('\r\n').split(' ')[1] for x in content]
        with open(out_file, 'w') as f2:
            for idx, line in enumerate(classes):
                f2.write("{} {}\n".format(idx+1, line))


def create_translation_dict(fname):
    with open(fname) as f:
        content = f.readlines()
        labels = [x.strip('\r\n').split(' ')[0] for x in content]
        newi_to_oldi = {}
        for idx, label in enumerate(labels):
            newi_to_oldi[idx+1] = label
        oldi_to_newi = {v: k for k, v in newi_to_oldi.iteritems()}
    return newi_to_oldi, oldi_to_newi


newi_to_oldi, oldi_to_newi = create_translation_dict('/hdd/NLN/UCF_list/truncated_classInd.txt')
# create_new_class_idx('/hdd/NLN/UCF_list/truncated_classInd.txt', '/hdd/NLN/UCF_list/classInd.txt')
# write_train_set('/hdd/NLN/UCF_list/trainlist00.txt', idx_to_class, '/hdd/NLN/UCF_list/trainlist04.txt')
# write_test_set('/hdd/NLN/UCF_list/testlist00.txt', class_to_idx, '/hdd/NLN/UCF_list/testlist04.txt')
convert_train_set('/hdd/NLN/UCF_list/trainlist04.txt', oldi_to_newi, '/hdd/NLN/UCF_list/trainlist05.txt')






