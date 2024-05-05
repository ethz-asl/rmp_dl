import collections
from contextlib import contextmanager
import copy
from multiprocessing import Manager, Process
import threading
from typing import OrderedDict
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
from dash import dash_table
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from queue import Empty, Queue

@contextmanager
def nonblocking(lock):
    locked = lock.acquire(False)
    try:
        yield locked
    finally:
        if locked:
            lock.release()


class VerticalParamList(OrderedDict): ...

def recursive_defaultdict():
    return collections.defaultdict(recursive_defaultdict)

class DashWrapper:
    app = None
    queue = None
    data_lock = threading.Lock()
    # Nested dict of {entity: {section: {name: data}}}
    data = recursive_defaultdict()

    @staticmethod
    def run_dash(queue):
        DashWrapper.queue = queue

        DashWrapper.app = dash.Dash(__name__)
        app = DashWrapper.app

        app.layout = html.Div([
            html.Div(id='dynamic-content'),
            dcc.Interval(id='page-update-interval', interval=100)  # Refresh the page content every second
        ])

        # Single callback to refresh the entire page content
        @app.callback(
                Output('dynamic-content', 'children'), 
                Input('page-update-interval', 'n_intervals'))
        def update_page(n_intervals):
            lock = DashWrapper.data_lock
            queue = DashWrapper.queue
            data = DashWrapper.data
            with nonblocking(lock) as locked:
                if locked and not queue.empty():
                    while not queue.empty():
                        # queue.empty does not guarantee an empty queue, 
                        # so put a 100ms timeout
                        try:
                            entity, section, name, d = queue.get(block=True, timeout=0.1) 
                        except Empty:
                            break
                        data[entity][section][name] = copy.deepcopy(d)
                    
                    return DashWrapper.get_all_children_from_data()
                else: 
                    return dash.no_update  

        app.run_server(debug=True, use_reloader=False)

    @staticmethod
    def get_all_children_from_data():
        children = []
        for entity in DashWrapper.data.keys():
            children.extend(DashWrapper.get_entity_section(entity))
        return children

    @staticmethod
    def get_entity_section(entity):
        children = [html.H3(f"{entity}")]

        sections = []
        for section in DashWrapper.data[entity].keys():
            content = DashWrapper.get_section_data(entity, section)
            width_style = {'display': 'inline-block'}
            sections.append(html.Div(content, style=width_style))
        
        children += reversed(sections)

        return [html.Div(children, style={'margin-bottom': '20px'})]

    @staticmethod
    def get_section_data(entity, section):
        return DashWrapper.get_children_from_data(DashWrapper.data[entity][section])


    @staticmethod
    def get_children_from_data(data):
        children = []
        for name, items in data.items():
            children.append(html.H5(name, style={'margin-bottom': '-20px'}))  # Reduced bottom margin

            row_children = []

            # We put multiple items on a row
            if not isinstance(items, list):
                items = [items]
            for item in items:
                if isinstance(item, np.ndarray):
                    table = DashWrapper.format_matrix(item)
                    row_children.append(html.Div([table], style={'display': 'inline-block', 'marginRight': '20px'}))  # Wrap in Div for inline display
                elif isinstance(item, str):  
                    text_element = DashWrapper.format_string(item)
                    row_children.append(text_element)
                elif isinstance(item, VerticalParamList):
                    element = DashWrapper.format_vertical_param_list(item)
                    row_children.append(element)

            children.append(html.Div(row_children, style={
                'display': 'flex',
                'alignItems': 'center',
                'marginBottom': '20px'
                }))  # Append the row to the main children list

        return children

    @staticmethod
    def format_vertical_param_list(param_list):
        elements = []
        for key, value in param_list.items():
            if isinstance(value, float):
                v = f'{value:.3f}'
            elif isinstance(value, int):
                v = f'{value}'
            elif isinstance(value, str):
                v = value
            elif isinstance(value, bool):
                v = 'True' if value else 'False'
            
            elements.append(html.Div([
                html.Span(key, style={'fontWeight': 'bold'}),
                html.Span(f'{value:.3f}', style={'marginLeft': '10px'})
            ]))
        return html.Div(elements, style={'marginTop': '40px', 'marginBottom': '20px'})   

    @staticmethod
    def format_string(item):
        text_element = html.Div(
            [html.Span(
                dcc.Markdown(item, dangerously_allow_html=True),
                style={
                    'fontSize': '16px',
                    'margin-top': '30px'
                })
             ],
            style={
                'display': 'inline-flex',
                'verticalAlign': 'middle',
                'alignItems': 'center',
                'marginRight': '20px'
            })
        return text_element

    @staticmethod
    def format_matrix(matrix):
        df = pd.DataFrame(matrix)

        max_abs_value = np.max(np.abs(matrix))

        # You can change 'viridis' to any other colormap
        cmap = plt.cm.get_cmap('YlOrRd')

        # Function to map value to color
        def value_to_color(value):
            # Normalize the value between 0 and 0.7, above 0.7 the color is too dark
            normalized_value = abs(value) / max_abs_value * 0.7 if abs(value) > 0.001 else 0
            rgba_color = cmap(normalized_value)  # Get RGBA color from colormap
            # Convert RGBA to hexadecimal
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(rgba_color[0]*255), int(rgba_color[1]*255), int(rgba_color[2]*255))
            return hex_color

        # Generate the style_data_conditional list
        style_data_conditional = []
        for col in df.columns:
            for row in range(len(df)):
                style_data_conditional.append({
                    'if': {'row_index': row, 'column_id': str(col)},
                    'backgroundColor': value_to_color(df.iloc[row, col]),
                    'color': 'black'
                })

        # Calculate table width based on number of columns
        cell_width = 100  # Fixed cell width in pixels
        table_width = len(df.columns) * cell_width

        table = dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{"name": str(i), "id": str(i), "type": "numeric", "format": {
                "specifier": ".3f"}} for i in df.columns],
            style_cell={
                'textAlign': 'center',
                'width': f'{cell_width}',  # Fixed cell width
                # Minimum width; ensures cell doesn't get smaller
                'minWidth': f'{cell_width}px',
                # Maximum width; ensures cell doesn't get bigger
                'maxWidth': f'{cell_width}px',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
            },
            style_table={
                # Set table width based on number of columns
                'width': f'{table_width}px',
                # 'minWidth': '100%',  # Ensures table takes up at least 100% of container width if smaller
            },
            style_header={'display': 'none'},  # Turning off column headers
            style_data_conditional=style_data_conditional
        )
        return table


class DashApp:
    manager = None
    sections = {}
    queue: Queue = None

    @staticmethod
    def run():
        DashApp.manager = Manager()
        DashApp.queue = DashApp.manager.Queue()

        dash_process = Process(target=DashWrapper.run_dash, args=(DashApp.queue, ))
        dash_process.start()
        return dash_process

    @staticmethod
    def update_data(entity, section, name, data):
        try:
            DashApp.queue.put((entity, section, name, data))
        except Exception as e:
            pass # If dashapp is uninitialized we just do nothing


if __name__ == '__main__':
    p = DashApp.run()
    DashApp.update_data('e1', 'left', 'matrix1', np.random.rand(5, 5))
    DashApp.update_data('e1', 'right', 'matrix1', np.random.rand(5, 5))
    DashApp.update_data('e2', 'left', 'matrix1', np.random.rand(5, 5))
    DashApp.update_data('e2', 'right', 'matrix1', np.random.rand(5, 5))

    p.join()
