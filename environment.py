from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import time
import numpy as np
import xml.etree.ElementTree as ET
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

app = Flask(__name__)
socketio = SocketIO(app)


model = AutoPeftModelForCausalLM.from_pretrained(
    'simple_model/checkpoint-60',
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
)
tokenizer = AutoTokenizer.from_pretrained('simple_model/checkpoint-60')

current_position = [0, 0]
current_color = 'grey'
current_obj = ''
flag_work = True

locations = {'table': {'x':5,'y':7}, 'bed': {'x':8,'y':4}, 'fridge': {'x':2,'y':2}, 'oven': {'x':9,'y':1}, 'door': {'x':0,'y':4}, 'window': {'x':1,'y':8}, 'HOME': {'x':0,'y':0}}

def fill_svg(x, y, color, command, thing):
    return f"""
            <!-- Room boundaries -->
            <rect x="0cm" y="0cm" width="12cm" height="12cm" fill="lightgray" />

            <!-- Fridge -->
            <rect x="2cm" y="2cm" width="1cm" height="2cm" fill="blue" />
            <text x="2cm" y="3cm" font-family="Arial" font-size="15" fill="black">Fridge</text>

            <!-- Bed -->
            <rect x="8cm" y="4cm" width="3cm" height="4cm" fill="orange" />
            <text x="8cm" y="5cm" font-family="Arial" font-size="15" fill="black">Bed</text>

            <!-- Table -->
            <rect x="5cm" y="7cm" width="2cm" height="1cm" fill="brown" />
            <text x="5cm" y="8cm" font-family="Arial" font-size="15" fill="black">Table</text>

            <!-- Oven -->
            <rect x="9cm" y="1cm" width="1cm" height="1cm" fill="red" />
            <text x="9cm" y="2cm" font-family="Arial" font-size="15" fill="black">Oven</text>

            <!-- Window -->
            <rect x="1cm" y="8cm" width="1cm" height="1cm" fill="violet" />
            <text x="1cm" y="9cm" font-family="Arial" font-size="15" fill="black">Window</text>

            <!-- Door -->
            <rect x="0cm" y="4cm" width="1cm" height="2cm" fill="green" />
            <text x="0cm" y="5cm" font-family="Arial" font-size="15" fill="black">Door</text>

            <!-- Robot -->
            <circle cx="{x}cm" cy="{y}cm" r="0.5cm" fill="grey" />

            <!-- Object -->
            <circle cx="{x}cm" cy="{y}cm" r="0.2cm" fill="{color}" />

            <text x="{x}cm" y="{y}cm" font-family="Arial" font-size="15" fill="black">{thing}</text>
            <text x="0cm" y="11.5cm" font-family="Arial" font-size="15" fill="black">{command}, current position: {x}, {y}</text>
        """

def get_response(task):
    prompt = f"""
    Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    You should generate XML to perform actions using these tags: task, action, actionType, object, location. Coordinates of locations: table, bed, fridge, oven, door, window.

    ### Input:
    {task}

    ### Response:
    """

    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
    outputs = model.generate(input_ids=input_ids, max_new_tokens=512, do_sample=True, top_p=0.6,temperature=0.9)

    return tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]

def parse_xml(bt):
    global current_position
    global current_color
    global current_obj
    tree = ET.ElementTree(ET.fromstring(bt))
    root = tree.getroot()
    print(format_element(root))
    emit('update_bt', format_element(root).replace('<', '&lt;').replace('>', '&gt;'), broadcast=True)
    states = []
    for child in root:
        if child[0].text == 'GO':
            location = locations[child[1].text]
            current_position[0] = location['x']
            state = {'x': current_position[0], 'y': current_position[1], 'color': current_color, 'command': 'GO', 'obj': current_obj}
            states.append(state)
            current_position[1] = location['y']
            state = {'x': current_position[0], 'y': current_position[1], 'color': current_color, 'command': 'GO', 'obj': current_obj}
            states.append(state)
        elif child[0].text == 'TAKE':
            current_color = 'red'
            current_obj = child[1].text
            state = {'x': current_position[0], 'y': current_position[1], 'color': current_color, 'command': 'TAKE', 'obj': current_obj}
            states.append(state)
        elif child[0].text == 'PUT':
            current_color = 'grey'
            current_obj = ''
            state = {'x': current_position[0], 'y': current_position[1], 'color': current_color, 'command': 'PUT', 'obj': current_obj}
            states.append(state)
    return states

def format_element(element, level=0):
    indent = '\t' * level
    if element.text is not None and element.text.strip():
        result = f'{indent}<{element.tag}>{element.text.strip()}'
    else:
        result = f'{indent}<{element.tag}>\n'

    for child in element:
        result += format_element(child, level + 1)

    result += f'{indent}</{element.tag}>\n'
    return result


@app.route('/')
def index():
    return render_template('index_socketio_play.html')

@socketio.on('update_svg')
def handle_update_svg(data):
    global flag_work
    task = data.get('input', None)
    if task.replace(' ', '') == '':
        emit('update_bt', 'Task is empty', broadcast=True)
    elif flag_work:
        flag_work = False
        response = get_response(task)
        states = parse_xml(response)
        print(states)
        for state in states:
            emit('svg_update_play', fill_svg(state['x'], state['y'], state['color'], state['command'], state['obj']), broadcast=True)
            time.sleep(1)
        flag_work = True
    else:
         emit('update_bt', 'Wait...', broadcast=True)

if __name__ == '__main__':
    socketio.run(app, port=5001, host='0.0.0.0')