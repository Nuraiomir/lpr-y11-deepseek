"""
Обучение детектора номерных знаков: YOLO11n, экспорт в ONNX 512x512.

Это ЕДИНСТВЕННЫЙ способ получить модель для этого сервиса. Сервис грузит
готовый .onnx и ничего не обучает, а веса в репозитории не хранятся: модель
здесь меняется, и каждая её версия в истории git раздувала бы репозиторий без
пользы.

ДВА ПРАВИЛА, КОТОРЫЕ НЕЛЬЗЯ НАРУШАТЬ.

Первое: экспорт ровно в 512. app/lpr/detector.py вписывает кадр в квадрат 512
и умножает нормализованный выход модели на 512. Экспорт другого размера не
упадёт, а вернёт НЕПРАВИЛЬНЫЕ рамки, и заметить это по поведению сервиса
почти невозможно: номера будут находиться, но не там, где они есть. Поэтому
скрипт отказывается от любого другого значения.

Второе: тот же датасет и то же разбиение при сравнении двух обучений. Иначе
измеряется датасет, а не обучение.

ЧТО НАДО ПРОВЕРИТЬ ДО ОБУЧЕНИЯ, и это не формальность. Кадры из видео, по
которым считается сквозное качество, в обучающем наборе лежать не должны.
Если лежат, качество будет расти на бумаге, пока сервис не станет хуже: модель
меряется на том, что запомнила. У нас это уже случилось: 150 кадров
проверочного видео лежали в обучении, и сквозная цифра была завышена.
Проверка делается инструментом dataset_overlap.py из исследовательского
репозитория, до запуска обучения, а не после.

ЧТО ДЕЛАТЬ ПОСЛЕ ОБУЧЕНИЯ. Порог уверенности СВОЙ У КАЖДЫХ ВЕСОВ: с новыми
весами меняется распределение уверенности, и прежний порог становится
случайным числом. Измерить его надо стендом detector_conf_sweep.py на
отложенной выборке и прописать в deployment/values как LPR_DETECT_CONF. См.
training/README.md.

ЗАВИСИМОСТИ. Нужна Ultralytics, а она под лицензией AGPL-3.0. В окружении
сервиса её нет и быть не должно: обучение и экспорт происходят здесь, а сервис
только грузит готовый .onnx. Ставится отдельно:

    python3.12 -m venv .venv_train
    .venv_train/bin/pip install -r training/requirements.txt

ЗАПУСК.

    .venv_train/bin/python training/train.py \\
        --data ~/training/lpr_kz/data.yaml --epochs 100
"""
import argparse
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMGSZ = 512  # не значение по умолчанию: см. шапку файла.


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path,
                    help="data.yaml обучающего набора")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--weights", default="yolo11n.pt",
                    help="начальные веса, скачиваются при первом запуске")
    ap.add_argument("--name", default="lpr_yolo11n",
                    help="имя прогона внутри runs/detect/")
    ap.add_argument("--imgsz", type=int, default=IMGSZ,
                    help=f"оставлен для наглядности, принимается только {IMGSZ}")
    ap.add_argument("--device", default="0")
    # У каждого обучения свой .onnx. Общее имя однажды позволило второму
    # прогону затереть первый, и сравнить их стало невозможно.
    ap.add_argument("--out-name", default="best11n_512.onnx",
                    help="имя файла в model/, куда положить экспорт")
    args = ap.parse_args()
    if not args.out_name.endswith(".onnx"):
        raise SystemExit(f"--out-name должен заканчиваться на .onnx: {args.out_name}")

    if args.imgsz != IMGSZ:
        raise SystemExit(
            f"--imgsz {args.imgsz} нельзя: сервис вписывает кадр в {IMGSZ} и "
            f"умножает выход модели на {IMGSZ}. Экспорт другого размера не упадёт, "
            f"а вернёт неправильные рамки, и это будет незаметно.")

    data = args.data.expanduser().resolve()
    if not data.is_file():
        raise SystemExit(f"нет файла с описанием датасета: {data}")

    from ultralytics import YOLO  # imported late: the error above is cheaper

    print(f"датасет: {data}")
    print(f"старт с весов: {args.weights}")
    print(f"эпох: {args.epochs}, размер: {IMGSZ}, батч: {args.batch}\n")

    t0 = time.perf_counter()
    model = YOLO(args.weights)
    model.train(data=str(data), epochs=args.epochs, imgsz=IMGSZ,
                batch=args.batch, device=args.device, name=args.name,
                project=str(ROOT / "runs" / "detect"))
    train_min = (time.perf_counter() - t0) / 60.0

    # Проверка на той части, которая названа в data.yaml, причём весами,
    # перечитанными с диска: тем самым файлом, который сейчас экспортируется.
    best = Path(model.trainer.best)
    metrics = YOLO(str(best)).val(data=str(data), imgsz=IMGSZ, device=args.device)

    onnx_tmp = Path(YOLO(str(best)).export(
        format="onnx", imgsz=IMGSZ, opset=12, simplify=True, dynamic=False))
    target = ROOT / "model" / args.out_name
    if target.exists():
        raise SystemExit(
            f"{target} уже есть. Перезаписать значит потерять модель, с которой "
            f"сравнивают. Задай другое --out-name или убери файл сам.")
    shutil.copy2(onnx_tmp, target)

    summary = {
        "weights": str(best),
        "onnx": str(target),
        "data": str(data),
        "epochs": args.epochs,
        "imgsz": IMGSZ,
        "train_minutes": round(train_min, 1),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "map50": round(float(metrics.box.map50), 4),
        "map50_95": round(float(metrics.box.map), 4),
    }
    summary["starting_weights"] = args.weights
    # По этому полю обучение потом связывается со своим экспортом: прогон
    # назван --name, файл назван --out-name, и больше их ничто не связывает.
    summary["out_name"] = target.name
    out = ROOT / "runs" / f"train_{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 62)
    print("ОБУЧЕНИЕ ЗАКОНЧЕНО")
    print("=" * 62)
    print(f"  время обучения   {summary['train_minutes']} мин")
    print(f"  точность         {summary['precision']:.3f}")
    print(f"  полнота          {summary['recall']:.3f}")
    print(f"  mAP50            {summary['map50']:.3f}")
    print(f"  mAP50-95         {summary['map50_95']:.3f}")
    print(f"\n  модель: {target}")
    print(f"  сводка: {out}")
    print("\nЭто метрики ДЕТЕКТОРА на отложенной выборке. Они НЕ говорят,")
    print("сколько машин назовёт сервис: рамка может найтись, а номер не")
    print("прочитаться. И они ничего не говорят о том, какой порог")
    print("уверенности правильный для этих весов.")
    print("\nДВА ОБЯЗАТЕЛЬНЫХ ШАГА ДАЛЬШЕ, см. training/README.md:")
    print("  1. Измерить порог уверенности на отложенной выборке и")
    print("     прописать его в deployment/values как LPR_DETECT_CONF.")
    print("     Прежний порог с новыми весами это случайное число.")
    print("  2. Прогнать сквозную проверку по видео: только она говорит,")
    print("     сколько машин сервис назовёт правильно.")


if __name__ == "__main__":
    main()
