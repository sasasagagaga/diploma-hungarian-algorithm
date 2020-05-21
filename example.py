import argparse
import warnings

import torch
import numpy as np

from Code import core, losses, style_transfer as ST, image_utils as IU


# TODO: добавить больше всяких сообщений для пользователя. Например, написать: "Стилизация будет проводится на CPU".


if __name__ == '__main__':
    # Папка для сохранения результатов
    save_dir = 'Results'

    # Нужно задать заранее для перечисления вариантов в хелпе.
    loss_name2loss_func = {
        'linear': (losses.LinearLoss, ST.linear_style_loss_weight_callback),
        'polynomial': (losses.PolynomialLoss, ST.polynomial_style_loss_weight_callback),
        'bn': (losses.BNLoss, ST.BN_style_loss_weight_callback),
        'gatys': (losses.GatysLoss, ST.gatys_style_loss_weight_callback),
        'hungarian': (losses.HungarianLoss, ST.hungarian_style_loss_weight_callback)
    }

    style_loss_vars = list(loss_name2loss_func.keys())

    # Парсинг аргументов
    class CustomFormatter(argparse.HelpFormatter):  # argparse.RawDescriptionHelpFormatter,
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)  # max_help_position=6, width=80)

        def _get_help_string(self, action):
            help = action.help
            if '%(default)' in action.help or action.default is argparse.SUPPRESS:
                return help

            defaulting_nargs = [argparse.OPTIONAL, argparse.ZERO_OR_MORE]
            if action.option_strings or action.nargs in defaulting_nargs:
                help += ' Значение по умолчанию: %(default)s.'
            return help


    # formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position=6, width=80)
    formatter_class = lambda prog: CustomFormatter(prog, max_help_position=6, width=80)

    parser = argparse.ArgumentParser(
        description='Алгоритм переноса стиля.',
        formatter_class=formatter_class,
        # formatter_class=argparse.ArgumentDefaultsHelpFormatter
        # formatter_class=CustomFormatter,
        add_help=False
    )

    def check_style_content(var):
        try:
            int(var)
        except ValueError:
            return var
        return int(var)

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                        help='Показать это сообщение с помощью и выйти.')
    parser.add_argument('-s', '--style', default=19, type=check_style_content,
                        help=f'Стилевое изображение. Либо название изображения из папки со стилями, либо индекс '
                             f'файла в папке со стилями (в этом случае может принимать значения от 0 до '
                             f'{core.num_styles - 1} включительно).')
    parser.add_argument('-c', '--content', default=3, type=check_style_content,
                        help='Контентное изображение. Либо название изображения из папки с контентами, либо индекс '
                             f'файла в папке с контентами (в этом случае может принимать значения от 0 до '
                             f'{core.num_contents - 1} включительно).')
    parser.add_argument('-ri', '--rand_init', action='store_true',
                        help='Использовать случайную начальную инициализацию для стилизации (по умолчанию в качестве '
                             'начальной инициализации для стилизации используется контентное изображение).')
    parser.add_argument('-sl', '--style_layers', default='all', type=str,
                        help='Слои, на которых будут считаться стилевые функции потерь. Нужно просто в строке '
                             'перечислить слитно номера нужных слоев. Номера могут быть от 1 до 5 включительно. '
                             'Также можно указать all, в этом случае будут использованы слои 1-5. Еще можно указать '
                             'no, в этом случае для стилизации вообще не будут использоваться слои.')
    parser.add_argument('-cl', '--content_layer', default='all',
                        help='Слой, на котором будет считаться контентная функция потерь. Нужно просто в строке '
                             'указать номер нужного слоя. '
                             'Также можно указать all, в этом случае будет использован слой 4. Еще можно указать '
                             'no, в этом случае для контента вообще не будут использоваться слои.')
    parser.add_argument('-sL', '--style_loss', dest='style_loss_class', default='gatys', choices=style_loss_vars,
                        help=f'Функция потерь для стиля. Может принимать одно из {len(style_loss_vars)} '
                             f'значений: {", ".join(style_loss_vars)}.')
    parser.add_argument('-S', '--suboptimal', dest='use_suboptimal', action='store_true',
                        help='Использовать субоптимальную версию венгерского алгоритма. Имеет '
                             'смысл только при выборе венгерского алгоритма в качестве функции потерь.')
    parser.add_argument('-ki', '--kmeans_init', default='random', type=str, choices=['random', 'k-means++'],
                        help='Инициализация для K-Means. Может быть либо "random", либо "k-means++". '
                             'Имеет смысл только при выборе венгерского алгоритма в качестве функции потерь.')
    parser.add_argument('-kmi', '--kmeans_max_iter', default=50, type=int,
                        help='Количество итераций для K-Means. Имеет смысл только при выборе '
                             'венгерского алгоритма в качестве функции потерь.')
    parser.add_argument('-knc', '--kmeans_n_clusters', default=80, type=int,
                        help='Количество кластеров для K-Means. Имеет смысл только при выборе '
                             'венгерского алгоритма в качестве функции потерь.')
    parser.add_argument('-pc', '--poly_c', default=0.0, type=float, help='Константа c для полиномиального ядра.')
    parser.add_argument('-mlc', '--measure_losses_contribution', action='store_true',
                        help='Делать на каждой итерации замеры вкладов функций потерь.')
    parser.add_argument('-eq', '--equalize', action='store_true',
                        help='Уравнивать силы наложения стиля и контента.')
    parser.add_argument('-ss', '--stylization_strength', default=1.5, type=float,
                        help='Сила наложения стиля, положительное число.')
    parser.add_argument('-aws', '--auto_weights_selection', action='store_true',  # dest='wanted_style_losses_contrib'
                        help='Включить автоматический подбор весов перед стилевыми функциями потерь.')
    parser.add_argument('-tvlw', '--tv_loss_weight', default=100.0, type=float,
                        help='Вес для total variation функции потерь, неотрицательное число.')
    parser.add_argument('-ni', '--num_iters', default=500, type=int,
                        help='Количество итераций для стилизации. Для алгоритма Гатиса рекомендуется 500 итераций, '
                             'для венгерского алгоритма  - 350.')
    parser.add_argument('-pi', '--print_iter', default=50, type=int,
                        help='Частота вывода информации о процессе стилизации на экран.')
    parser.add_argument('-tpo', '--turn_print_off', action='store_true',
                        help='Полностью отключить вывод информации о процессе стилизации на экран.')
    parser.add_argument('-si', '--show_iter', default=False, type=int,
                        help='Частота вывода текущего изображения стилизации на экран.')
    parser.add_argument('-tso', '--turn_show_off', action='store_true',
                        help='Полностью отключить вывод текущего изображения стилизации на экран.')
    parser.add_argument('-v', '--verbose', default=0, type=int, help='Сила логирования процесса переноса стиля.')
    parser.add_argument('-of', '--out_file', default='result.jpg', type=str,
                        help=f'Название файла, в который будет сохранена стилизация. '
                             f'Все стилизации сохраняются в папке {save_dir}')
    parser.add_argument('-sg', '--save_gif', action='store_true',
                        help='Сохранить gif-изображение процесса переноса стиля.')
    parser.add_argument('-gf', '--gif_file', default='result.gif', type=str,
                        help=f'Название файла, в который будет сохранено gif-изображение. '
                             f'Все изображения сохраняются в папке {save_dir}')
    parser.add_argument('-cpu', '--use_cpu', action='store_true', help=f'Использовать только CPU во время стилизации.')
    parser.add_argument('-gpu', '--gpu_number', default=0, type=int,
                        help=f'Номер GPU, на котором проводить стилизацию.')
    parser.add_argument('-is', '-size', '--img_size', default=300, type=int,
                        help=f'Размер минимальной стороны изображения, целое число.')

    # DONE: задать еще аргумент для выходных файлов           -> сделал для стилизации. Для гифки надо?
    # DONE: добавить перечисление всех вариантов для style_loss и мб для каких-то еще.   -> choices
    # DONE: добавить кастомный размер изображения, возможность использования ЦПУ и т.п.
    # TODO: добавить возможность "достилизовать" изображение (продолжить стилизацию).
    # DONE: изменить ширину хелпа. И сам текст хелпа.
    # DONE: опции для полного отключения логирования.

    args = parser.parse_args()

    # Обрабатываем эквализацию
    if not args.rand_init and args.equalize:
        warnings.warn('При эквализации силы стиля и контента стоит начинать стилизацию со случаной инициализации.\n')

    if not args.equalize:
        args.stylization_strength = None

    # Обрабатываем аргументы для стилевых функций потерь
    style_loss_kwargs = {}
    if args.style_loss_class == 'hungarian':
        style_loss_kwargs = {
            'use_suboptimal': args.use_suboptimal,
            'kmeans_kwargs': {
                'init': args.kmeans_init,
                'max_iter': args.kmeans_max_iter,
                'n_clusters': args.kmeans_n_clusters,
            }
        }
    elif args.style_loss_class == 'linear':
        style_loss_kwargs = {
            'c': args.poly_c
        }

    # Обрабатываем стилевые функции потерь
    args.style_loss_class, style_loss_weight_callback = loss_name2loss_func[args.style_loss_class.lower()]
    if args.auto_weights_selection:
        style_loss_weight_callback = 'auto'

    # Обрабатываем флаги для логирования
    if args.turn_print_off:
        args.print_iter = False
    if args.turn_show_off:
        args.show_iter = False

    # Обрабатываем core переменные
    core.img_size = args.img_size
    if args.use_cpu or not torch.cuda.is_available():
        core.device = torch.device('cpu')
    else:
        core.device = torch.device(f'cuda:{args.gpu_number}')


    # import pprint
    # pprint.pprint(vars(args))
    # print(style_loss_weight_callback)


    # Перенос стиля
    result = ST.full_style_transfer(
        [args.style, args.content],
        dict(
            opt_img_rand=args.rand_init
            # Если нужно эквализировать силу наложение стиля и контента, то
            # надо начинать из случайного приближения.
        ), dict(
            style_layers=args.style_layers,
            content_layers=args.content_layer,
            style_loss_class=args.style_loss_class,
            style_loss_weight_callback=style_loss_weight_callback,
            **style_loss_kwargs
        ), dict(
            measure_losses_contribution=args.measure_losses_contribution,
            equalize_content_and_style=args.equalize,
            stylization_strength=args.stylization_strength,
            wanted_style_losses_contrib=args.auto_weights_selection,

            tv_loss_weight=args.tv_loss_weight,
            max_iter=args.num_iters,
            print_iter=args.print_iter,
            show_iter=args.show_iter,
            return_gif=args.save_gif,
            verbose=args.verbose,
        )
    )

    # Сохранение результатов стилизации
    np.save(f'{save_dir}/results', result)

    if '.' not in args.out_file:
        args.out_file = f'{args.out_file}.jpg'
    print(f'{save_dir}/{args.out_file}')
    result.opt_img.save(f'{save_dir}/{args.out_file}')

    if args.save_gif:
        if args.gif_file.endswith('.gif'):
            args.gif_file = args.gif_file[:-4]
        IU.save_gif(result.images, args.gif_file, save_dir)
