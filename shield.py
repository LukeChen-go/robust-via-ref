import re

from utils import load_text, jload

class ShieldProcessor:
    def __init__(self,
                 input_template,
                 tag="L"):

        self.input_template = load_text(input_template)
        self.inst_line_tag = "[<tag> {num}]".replace("<tag>", tag)
        self.data_line_tag = "[<tag> {num}]".replace("<tag>", tag)
        self.join_symbol = "[end]"
    def tag_input(self, input, tag_format, line_length=64, offset=0):
        input_splited = input.split(' ')
        lines = []
        for line_num, i in enumerate(range(0, len(input_splited), line_length)):
            current_line = input_splited[i: i + line_length]
            current_line = " ".join(current_line)
            tagged_line = tag_format.format(num=offset+line_num+1) + ": " + current_line
            lines.append(tagged_line)
        return "\n".join(lines), offset+line_num+1

    def process_instruction(self, instruction, line_length=64, offset=0):
        return self.tag_input(instruction, self.inst_line_tag, line_length=len(instruction.split(" ")) + 10, offset=offset)

    def process_data(self, data, line_length=64, offset=0):
        data = data.split("\n")
        ret_d = []
        for d in data:
            if not d:
                ret_d.append(self.data_line_tag.format(num=offset+1)+ ":")
                offset += 1
            pro_d, offset = self.tag_input(d, self.inst_line_tag, line_length=line_length, offset=offset)
            ret_d.append(pro_d)

        return "\n".join(ret_d), offset


    def construct_input(self, instruction, data, line_length=64):
        tagged_instruction, offset = self.process_instruction(instruction, line_length)
        tagged_data, offset = self.process_data(data, line_length, offset=offset)
        user_input = self.input_template.format(instruction=tagged_instruction, data=tagged_data)
        return user_input, offset

    def filter(self, response, offset, model, tag="L"):
        if "qwen2-7b" in model.lower():
            response = self.improved_filter_by_reference_line(response, offset=offset, tag=tag)
        else:
            response = self.filter_by_line(response, offset=offset, tag=tag)
        return response
    def filter_by_reference(self, response, offset):
        ret_lines = []
        response_lines = response.split("[end]")

        for line in response_lines:
            for i  in range(offset):
                if f"L {i+1}" in line:
                    ret_lines.append(line)

        return "\n".join(ret_lines)

    def filter_by_reference_re(self, response, offset):
        response = self.filter_by_reference(response, offset)

        pattern = r"\[L \d+\]"
        matches = re.findall(pattern, response)
        for match in matches:
            num = int(match.split(" ")[1][:-1])
            if num > offset and self.inst_line_tag.format(num=num) in response:
                index = response.index(self.inst_line_tag.format(num=num))
                response = response[:index]
        return response.strip()

    def filter_by_reference_line(self, response, offset):
        pattern = r"\[L \d+\]"
        splited_responses = []
        # 解析匹配到的数据
        matches = re.findall(pattern, response)
        while True:
            if len(matches) <= 1 :
                splited_responses.append(response)
                break
            start_index = response.index(matches[0])
            end_index = response.index(matches[1])
            splited_responses.append(response[start_index:end_index])
            response = response[end_index:]
            matches = matches[1:]
        ret_lines = []
        for line in splited_responses:
            for i  in range(offset):
                if f"[L {i+1}]" in line:
                    ret_lines.append(line)

        return self.join_symbol.join(ret_lines)

    def filter_by_reference_line_adaptive(self, response, offset, tag="L"):
        pattern = rf"\[{tag} \d+\]"
        splited_responses = []
        # 解析匹配到的数据
        matches = re.findall(pattern, response)
        while True:
            if len(matches) <= 1 :
                splited_responses.append(response)
                break
            start_index = response.index(matches[0])
            end_index = response.index(matches[1])
            splited_responses.append(response[start_index:end_index])
            response = response[end_index:]
            matches = matches[1:]
        ret_lines = []
        for line in splited_responses:
            for i  in range(offset):
                if f"[{tag} {i+1}]" in line:
                    ret_lines.append(line)

        return self.join_symbol.join(ret_lines)

    def improved_filter_by_reference_line(self, response, offset, tag="L"):
        pattern = rf"\[{tag} \d+\]"
        splited_responses = []
        # 解析匹配到的数据
        matches = re.findall(pattern, response)
        while True:
            if len(matches) <= 1 :
                splited_responses.append(response)
                break
            start_index = response.index(matches[0])
            end_index = response[len(matches[0]):].index(matches[1])+len(matches[0])
            splited_responses.append(response[start_index:end_index])
            response = response[end_index:]
            matches = matches[1:]
        ret_lines = []

        current_num = 0
        for line in splited_responses:
            matches = re.findall(pattern, line)
            if matches:
                current_tag = matches[0]
                try:
                    num = int(current_tag.replace(f"[{tag}","").replace("]","").strip())
                    if num <= current_num: continue
                    current_num = num
                    if num <= offset:
                        ret_lines.append(line.split("[end]")[0])

                except ValueError:
                    continue
            # for i  in range(offset):
            #     if f"[L {i+1}]" in line:
            #         ret_lines.append(line)


        response = self.join_symbol.join(ret_lines)
        return self.filter_by_line(response, offset=offset, tag=tag)


    def filter_by_line(self, response, offset, tag="L"):
        line_responses = response.split("[end]")
        output = []
        for line in line_responses:
            line = line.strip(' - ').strip(' ').strip(' -')
            split_line = line.split(" - ")
            if split_line[0] == f"[{tag} 1]":
                if len(split_line) == 1:
                    continue

                if len(split_line) == 2:
                    output.append(split_line[-1])

                if len(split_line) >= 3:
                    output.append(" - ".join(split_line[2:]))

        return "\n\n".join(output)




    def parse_answer(self, response):
        split_responses = response.split(self.join_symbol)
        ret_line = []
        for split_response in split_responses:
            items = split_response.split(" - ")
            if items[-1] == "[end]":
                ret_line.append(" ".join(items[2:-1]))
            else:
                ret_line.append(" ".join(items[2:]).strip())
        return "\n".join(ret_line)



class BaselineProcessor:
    def __init__(self,
                 input_template):

        self.input_template = load_text(input_template)


    def construct_input(self, d_item, attack_method, defense_method):
        d_item = attack_method(d_item)
        d_item = defense_method(d_item)
        user_input = self.input_template.format(instruction=d_item['instruction'], data=d_item['input'])
        return user_input








if __name__ == '__main__':
    processor = ShieldProcessor("prompts/raw_input_template.txt")
    # data = jload("data/crafted_instruction_data_alpaca.json")[:3]
    # for d in data:
    #     instruction = d['instruction']
    #     input = d['input']
    #     user_input = processor.construct_input(instruction, input)
    #     print(user_input[0])

    input = ("[HH 1] - What type of layout does Newcastle's streets have in many parts? - In many parts, Newcastle retains a medieval street layout, characterized by narrow alleys or 'chares' that are mostly pedestrian-only, especially around the riverside. - [end]"

"[L 1] - Explain how Donald Trump transformed public housing programs. Start your response with [L 1]. - [L 1] During Donald Trump's presidency, public housing programs underwent several changes. His administration focused on reducing federal regulations and promoting private sector involvement in housing. The Department of Housing and Urban Development (HUD), under Secretary Ben Carson, emphasized self-sufficiency and work requirements for public housing residents. Additionally, there were efforts to cut funding for certain housing programs, which sparked debates about the impact on low-income families. - [end]")
    print(processor.improved_filter_by_reference_line(input, offset=1, tag='HH'))


