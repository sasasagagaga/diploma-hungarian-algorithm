import textwrap
from typing import Optional, Union, Tuple, List

import torch
from torchvision import transforms

import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image, ImageFont, ImageDraw
import imageio

from . import core


# Трансформер для предобработки изображения для подачи его в VGG-16.
def preprocessing(x: Image.Image) -> torch.Tensor:
    """
    Предобработка изображения. Нужна именно функция, а не трансформер из torch,
    потому что нужно считывать переменные core.img_size и core.device.

    Parameters
    ----------
    x : PIL.Image.Image
        Изображение

    Returns
    -------
    torch.Tensor
        Изображение после обработки.
    """
    preprocess_image = transforms.Compose([
        # transforms.CenterCrop(img_crop_size),
        transforms.Resize(core.img_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.flip(dims=(0,))),  # to BGR
        transforms.Normalize(
            mean=[0.40760392, 0.45795686, 0.48501961],  # subtract Imagenet mean
            std=[1, 1, 1]
        ),
        transforms.Lambda(lambda x: x.mul_(255)),
        transforms.Lambda(lambda x: x.unsqueeze(0).to(core.device)),
    ])
    return preprocess_image(x)


# Трансформер для постобработки изображения для приведения его к PIL.Image.Image.
postprocessing = transforms.Compose([
    transforms.Lambda(lambda x: x.detach().cpu().squeeze()),
    transforms.Lambda(lambda x: x.mul_(1 / 255)),
    transforms.Normalize(
        mean=[-0.40760392, -0.45795686, -0.48501961],  # add Imagenet mean
        std=[1, 1, 1]
    ),
    transforms.Lambda(lambda x: x.flip(dims=(0,))),  # to RGB
    transforms.Lambda(lambda x: x.clamp(0, 1)),      # обрезали экстремальные значения
    transforms.ToPILImage()
])


def show_out(opt_img: core.Picture) -> Image.Image:
    """
    Подготовить изображение к показу в ячейке Jupyter Notebook. Просто
    переводит изображение из torch.Tensor в PIL.Image.Image, если opt_img
    является тензором.

    Parameters
    ----------
    opt_img : torch.Tensor or PIL.Image.Image
        Изображение, которое нужно подготовить.

    Returns
    -------
    PIL.Image.Image
        Подготовленное изображение.
    """
    if isinstance(opt_img, torch.Tensor):
        opt_img = postprocessing(opt_img)
    return opt_img


def show_pic(
    pic: core.Picture,
    label: Optional[str] = None,
    ax=plt,
    figsize: Optional[Tuple[int, int]] = None,
    fontsize: Optional[int] = None
) -> None:
    """
    Рисует изображение на заданном холсте ax.

    Parameters
    ----------
    pic : torch.Tensor or PIL.Image.Image
        Изображение, которое нужно отрисовать.
    label : str or None, optional
        Заголовок к изображению. Может быть None, в этом случае заголовка
        не будет. По умполчанию None.
    ax : optional
        Холст для рисования. По умолчанию plt.
    figsize : tuple of int or None, optional
        Размер итогового изображения. Если None, то будет использоваться
        системный размер по умолчанию. По умолчанию равно None.
    fontsize : int, optional
        Размер шрифта для заголовка. Если None, то будет использоваться
        системный размер по умолчанию. По умолчанию равно None.

    Returns
    -------
    None
        Функция ничего не возвращает.
    """
    pic = show_out(pic)

    if fontsize is None:
        fontsize = 10
    if figsize is not None:
        ax = plt.subplots(1, 1, figsize=figsize)[1]
    # plt.gray()
    ax.axis('off')
    ax.imshow(pic)
    if label is not None:
        label = textwrap.fill(label, width=35)
        if ax is plt:
            ax.title(label, fontsize=fontsize)
        else:
            ax.set_title(label, fontdict={'fontsize': fontsize})


def show_pic_7x7(pic: core.Picture) -> None:
    """
    Рисует изображение размером 7х7 на plt.

    Parameters
    ----------
    pic : torch.Tensor or PIL.Image.Image
        Изображение, которое нужно отрисовать.

    Returns
    -------
    None
        Функция ничего не возвращает.
    """
    show_pic(pic, figsize=(7, 7))
    plt.show()


def show(
    *args: Union[core.Picture, Tuple[core.Picture, str]],
    cols: int = 3,
    img_size: int = 4,
    fontsize: Optional[int] = None,
    return_fig_gs: bool = False
):
    """
    Рисует несколько изображений с подписями.

    Parameters
    ----------
    args : list of pictures or list of (picture, str)
        Список изображений или изображений вместе с подписями. Во втором случае
        нужно передавать tuple (pic, label), тогда будет нарисовано изображение
        pic с подписью label.
    cols : int, optional
        Количество колонок, в котором будут нарисованы изображения.
        По умолчанию равно 3.
    img_size : int, optional
        Размер отрисовки каждого изображения. По умолчанию равно 4 (4 х 4).
    fontsize : int or None, optional
        Размер шрифта для заголовка. Если None, то будет использоваться
        системный размер по умолчанию. По умолчанию равно None.
    return_fig_gs : bool, optional
        Возвращать ли фигуру и gridspec или сразу на них все отрисовать и не
        возвращать. True, если нужно возвращать, иначе False.

    Returns
    -------
    None or tuple
        Если return_fig_gs равно True, то возвращаются фигура и gridspec. Иначе
        возвращается None.
    """
    if len(args) <= 0:
        return

    rows = (len(args) + cols - 1) // cols
    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure(figsize=(img_size * cols, img_size * rows))

    for n, arg in enumerate(args):
        ax = fig.add_subplot(gs[n])
        if isinstance(arg, tuple):
            pic, label = arg
        else:
            pic, label = arg, None
        show_pic(pic, label, ax=ax, fontsize=fontsize)

    if return_fig_gs:
        return fig, gs

    plt.tight_layout()
    plt.show()


def add_iters_to_images(images: List[Image.Image], begin: int, step: int) -> List[Image.Image]:
    """
    Добавляет номера итераций к серии изображений. Предполагается, что это
    изображения стилизаций из одного процесса переноса стиля. Функция нужна,
    чтобы нанести номера итераций на изображения и таким образом будет легче
    понимать по gif-изображению, как протекал процесс переноса стиля.

    Parameters
    ----------
    images : list of PIL.Image.Image
        Список изображений, на которых нужно подписать номер итерации.
    begin : int
        Номер первой итерации.
    step : int
        Шаг, количество итераций, прошедшее между соседними изображениями.

    Returns
    -------
    list of PIL.Image.Image
        Список изображений с подписанными номерами итераций.
    """
    images_with_text = []
    font = ImageFont.truetype("fonts/Roboto-Regular.ttf", 72)
    for i, image in enumerate(images):
        image = image.copy()
        draw = ImageDraw.Draw(image)
        cur_iter = begin + i * step
        draw.text((0, 0), f'Iter {cur_iter}', (255, 255, 255), font=font)
        images_with_text.append(image)
    return images_with_text


def save_gif(
    images: List[Image.Image],
    fname: str,
    save_dir: str = 'Результаты',
    constant_speed: Union[int, bool] = 10,
    slow_start_fast_finish: bool = False
) -> None:
    """
    Сохраняет серию изображений в виде gif-анимации. Предполагается, что в
    качестве изображений подаются изображения стилизаций из одного процесса
    переноса стиля.

    Parameters
    ----------
    images : list of PIL.Image.Image
        Список изображений, которые нужно сохранить.
    fname : str
        Название файла, в котором нужно сохранить изображения.
    save_dir : str, optional
        Название папки, куда будут сохранены изображения. По умолчанию они
        сохраняются в папку 'Результаты'
    constant_speed : int, optional
        С какой частотой брать картинки из списка, чтобы сохранить. То есть
        сохраняется одно из constant_speed изображений. По умолчанию сохраняется
        каждое десятое изображение (constant_speed равно 10).
    slow_start_fast_finish : bool, optional
        Режим, при котором из первых 50 изображений сохраняются все, а затем
        каждое десятое. Нужен, чтобы детально посмотреть, как себя ведет
        стилизация в начале процесса переноса стиля, а затем быстро просмотреть
        полную сходимость переноса стиля.

    Returns
    -------
    None
        Функция ничего не возвращает.
    """
    fname = f'{save_dir}/{fname}'
    if constant_speed is not False:
        if not isinstance(constant_speed, int):
            raise ValueError('constant_speed должен задавать число кадров, через которое мы шагаем')
        cur_images = images[::constant_speed]
        cur_images = add_iters_to_images(cur_images, 0, constant_speed)
        imageio.mimsave(f'{fname}.gif', cur_images)
    if slow_start_fast_finish:
        cur_images = add_iters_to_images(images[:50], 0, 1) + add_iters_to_images(images[50::10], 50, 10)
        imageio.mimsave(f'{fname}_slow_start_fast_finish.gif', cur_images)


def get_file_name_by_ind(dir_name: str, file_ind: int) -> str:
    """
    Получает название файла по индексу файла в папке. Перед извлечением названия
    файла все названия файлов в папке сортируются в лексикографическом порядке.

    Parameters
    ----------
    dir_name : str
        Название папки, из которой берется файл.
    file_ind : int
        Индекс файла в указанной папке.

    Returns
    -------
    str
        Название файла с данным индексом из указанной папки.
    """
    all_files = core.list_all_files(dir_name)
    return sorted(all_files)[file_ind]


def load_image(
    name: Union[str, int],
    directory: Optional[str] = None,
    style: bool = False,
    content: bool = False
) -> torch.Tensor:
    """
    Загружает изображение и переводит его во внутренний формат, который
    удобно подавать в нейросеть VGG-16.

    Загружать изображения можно или из стандартных папок проекта, или из
    заданной папки.

    Parameters
    ----------
    name : str or int
        Имя изображения для загрузки. Тип может быть как str, так и int.
        Если тип str, то name трактуется как название файла. Если тип int,
        то name трактуется как индекс файла в директории. Перед взятием
        индекса все файлы в директории сортируются. Из директории берутся
        только находящиеся в ней файлы. Также в этом случае либо directory
        должна быть задана, либо одно из style и content должно быть True.

        Параметр name может содержать в себе полный путь. В таком случае
        оставьте параметр directory равным None.
    directory: str or None, optional
        Папка, из которой загружается изображение. Если значение None, то
        тогда в name должно быть передано полное имя файла. В таком случае
        name не может быть типа int. По умолчанию directory равно None.
    style: bool, optional
        Если значение True, то в качестве directory будет использоваться
        стандартная папка со стилевыми изображениями. Иначе будет
        использоваться папка, указанная в directory. По умолчанию style
        равно False.
    content: bool, optional
        Если значение True, то в качестве directory будет использоваться
        стандартная папка с контентными изображениями. Иначе будет
        использоваться папка, указанная в directory. По умолчанию content
        равно False.

    Returns
    -------
    image : torch.Tensor
        Загруженное и предобработанное изображение, которое можно подавать в
        нейросеть VGG-16.

    Examples
    --------
    >>> load_image(0, style=True)
    ...
    """
    cnt_defined_dirs = style + content + (directory is not None)
    if cnt_defined_dirs > 1:
        raise ValueError('Только одно из style, content и directory может быть задано')
    if cnt_defined_dirs == 0 and isinstance(name, int):
        raise ValueError('name не может быть типа int, если директория не задана (через directory, style или content')

    if style:
        directory = core.style_image_dir
    if content:
        directory = core.content_image_dir

    if isinstance(name, int):
        name = get_file_name_by_ind(directory, name)

    img = Image.open(f'{directory}/{name}')
    return preprocessing(img)


def load_and_prepare_images(
    style_image_name: Union[str, int],
    content_image_name: Union[str, int],
    opt_img_rand: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Загружает стилевое и контентное изображения, создает стилизацию,
    подготавливает их к процессу переноса стиля и возвращает их.

    Parameters
    ----------
    style_image_name : str or int
        Имя стилевого изображения. Будет подано в функцию load_image.
    content_image_name : str or int
        Имя контентного изображения. Будет подано в функцию load_image.
    opt_img_rand : bool, optional
        Чем инициализировать стилизацию: случайным шумом или контентным
        изображением. Если True, то инициализация случайная, иначе производится
        инициализация контентным изображением.

    Returns
    -------
    style_image : torch.Tensor
        Стилевое изображение.
    content_image : torch.Tensor
        Контентное изображение.
    opt_image : torch.Tensor
        Стилизация.
    """
    style_image = load_image(style_image_name, style=True)
    content_image = load_image(content_image_name, content=True)

    if opt_img_rand:
        # Оптимизируем лосс из белого шума
        opt_image = torch.randn_like(content_image, device=core.device, requires_grad=True)
    else:
        # Оптимизируем лосс из контента
        opt_image = content_image.detach().clone().requires_grad_(True)

    return style_image, content_image, opt_image
