import os
import argparse


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def main():
    assert os.path.exists(inputdir), 'Input dir not found'
    assert os.path.exists(targetdir), 'target dir not found'
    # mkdir(outputdir)
    imgs = sorted(os.listdir(inputdir))
    for idx,img in enumerate(imgs):
        groups = ''

        groups += os.path.join(inputdir, img) + '|'
        suffix, exp = os.path.splitext(img)
        img = img.replace(suffix,suffix.split('_')[0])
        img = img.replace('jpg','png')
        groups += os.path.join(targetdir,img)

        with open(output, 'a') as f:
            f.write(groups + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='', metavar='PATH', help='root dir of lq images')
    parser.add_argument('--target', type=str, default='', metavar='PATH', help='root dir of hq images')
    parser.add_argument('--output', type=str, default='./docs/train/GoPro.txt', metavar='PATH', help='output dir to save group txt files')
    parser.add_argument('--ext', type=str, default='.png', help='Extension of files')
    args = parser.parse_args()

    inputdir = args.input
    targetdir = args.target
    output = args.output
    filename = args.filename
    ext = args.ext

    main()
