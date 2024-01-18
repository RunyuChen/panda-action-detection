import re
import json

def update_av_attribute(json_file_path, output_file_path):
    # 定义前缀与动作类别的映射关系
    prefix_mapping = {
        '1-250': '2',
        '251-552': '0',
        '2000-2309': '1',
        'other': '3'
    }

    # 读取JSON文件内容
    with open(json_file_path, 'r') as file:
        json_data = json.load(file)

    # 获取文件和元数据信息
    files = json_data['file']
    metadata = json_data['metadata']

    # 遍历每张图片的元数据
    for image_key, image_metadata in metadata.items():
        # 提取文件编号
        file_number = re.search(r'\d+', image_key).group()

        # 根据文件编号确定前缀
        prefix = 'other'
        if 1 <= int(file_number) <= 250:
            prefix = '1-250'
        elif 251 <= int(file_number) <= 552:
            prefix = '251-552'
        elif 2000 <= int(file_number) <= 2309:
            prefix = '2000-2309'

        # 获取对应的动作类别
        action_category = prefix_mapping.get(prefix, '3')

        # 更新元数据中的 av 属性
        image_metadata['av']['1'] = action_category

    # 将更新后的JSON数据写入新的文件
    with open(output_file_path, 'w') as output_file:
        json.dump(json_data, output_file, indent=2)


input_json_path = '/root/autodl-tmp/1_proposal_s.json'
output_json_path = '/root/autodl-tmp/output.json'
update_av_attribute(input_json_path, output_json_path)
