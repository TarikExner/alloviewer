from .image_analysis.structs import PlateLayout
layout = PlateLayout(
    wells={
        'A1': 'negative', 'A2': 'sample',  'A3': 'sample',  'A4': 'sample',  'A5': 'sample',
        'A6': 'sample',   'A7': 'sample',  'A8': 'sample',  'A9': 'sample',  'A10': 'positive',

        'B1': 'negative', 'B2': 'sample',  'B3': 'sample',  'B4': 'sample',  'B5': 'sample',
        'B6': 'sample',   'B7': 'sample',  'B8': 'sample',  'B9': 'sample',  'B10': 'positive',

        'C1': 'sample',   'C2': 'sample',  'C3': 'sample',  'C4': 'sample',  'C5': 'sample',
        'C6': 'sample',   'C7': 'sample',  'C8': 'sample',  'C9': 'sample',  'C10': 'sample',

        'D1': 'sample',   'D2': 'sample',  'D3': 'sample',  'D4': 'sample',  'D5': 'sample',
        'D6': 'sample',   'D7': 'sample',  'D8': 'sample',  'D9': 'sample',  'D10': 'sample',

        'E1': 'sample',   'E2': 'sample',  'E3': 'sample',  'E4': 'sample',  'E5': 'sample',
        'E6': 'sample',   'E7': 'sample',  'E8': 'sample',  'E9': 'sample',  'E10': 'sample',

        'F1': 'sample',   'F2': 'sample',  'F3': 'sample',  'F4': 'sample',  'F5': 'sample',
        'F6': 'sample',   'F7': 'sample',  'F8': 'sample',  'F9': 'sample',  'F10': 'sample',
    }
)

image_order=[
    'A1', 'B1', 'C1', 'D1', 'E1', 'F1',  # col 1 (top → bottom)
    'F2', 'E2', 'D2', 'C2', 'B2', 'A2',  # col 2 (bottom → top)
    'A3', 'B3', 'C3', 'D3', 'E3', 'F3',  # col 3
    'F4', 'E4', 'D4', 'C4', 'B4', 'A4',  # col 4
    'A5', 'B5', 'C5', 'D5', 'E5', 'F5',  # col 5
    'F6', 'E6', 'D6', 'C6', 'B6', 'A6',  # col 6
    'A7', 'B7', 'C7', 'D7', 'E7', 'F7',  # col 7
    'F8', 'E8', 'D8', 'C8', 'B8', 'A8',  # col 8
    'A9', 'B9', 'C9', 'D9', 'E9', 'F9',  # col 9
    'F10', 'E10', 'D10', 'C10', 'B10', 'A10',  # col 10
]
