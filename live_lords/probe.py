"""Bounded developer probe; all coordinates use the normalized 1280×720 view."""
import argparse
import json
from pathlib import Path
from .driver import LDPlayer, OCR, DEFAULT_NAME

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=DEFAULT_NAME)
    parser.add_argument('--tap', nargs=2, type=int)
    parser.add_argument('--back', action='store_true')
    parser.add_argument('--ocr', action='store_true')
    args = parser.parse_args()
    driver = LDPlayer(Path(r'C:\LDPlayer\LDPlayer9'), args.name, Path('evidence/live-yumi'))
    image = driver.capture()
    if args.tap or args.back:
        if args.tap: driver.tap(*args.tap)
        else: driver.back()
        import time
        time.sleep(1)
        image = driver.capture()
    if args.ocr:
        result = OCR().read(driver.data / 'latest.png')
        (driver.data / 'ocr.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), 'utf-8')
        print(result['text'])

if __name__ == '__main__': main()
