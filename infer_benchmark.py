#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import time
import os
import subprocess
from collections import deque

import matplotlib.pyplot as plt


def play_audio(mp3_path, label, mute=False):
    if mute:
        print(f"[AUDIO muted] {label}")
        return
    if not os.path.exists(mp3_path):
        print(f"[WARN] áudio não encontrado: {mp3_path} ({label})")
        return
    try:
        # rasp: mpg123
        subprocess.Popen(["mpg123", "-q", mp3_path])
    except Exception as e:
        print(f"[WARN] falha tocando {mp3_path}: {e}")


def letterbox(img, new_shape):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    pad_h, pad_w = new_shape - nh, new_shape - nw
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return im, r, (left, top)


def run_inference(net, frame, imgsz, conf_thres, iou_thres):
    inp, r, (dx, dy) = letterbox(frame, imgsz)
    blob = cv2.dnn.blobFromImage(inp, 1 / 255.0, (imgsz, imgsz), swapRB=True, crop=False)
    net.setInput(blob)

    t0 = time.time()
    raw = net.forward()
    infer_ms = (time.time() - t0) * 1000.0

    out = np.squeeze(raw)
    # garantir que fique (N, 4+cls)
    if out.ndim == 2 and out.shape[0] < out.shape[1]:
        out = out.transpose(1, 0)

    boxes, scores, cids = [], [], []

    for det in out:
        cx, cy, w_box, h_box = det[0:4]
        cls_scores = det[4:]
        cid = int(np.argmax(cls_scores))
        score = float(cls_scores[cid])
        if score < conf_thres:
            continue

        x1 = (cx - w_box / 2 - dx) / r
        y1 = (cy - h_box / 2 - dy) / r
        x2 = (cx + w_box / 2 - dx) / r
        y2 = (cy + h_box / 2 - dy) / r

        boxes.append([int(x1), int(y1), int(x2), int(y2)])
        scores.append(score)
        cids.append(cid)

    # NMS
    nms_boxes = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes]
    idxs = cv2.dnn.NMSBoxes(nms_boxes, scores, conf_thres, iou_thres)

    f_boxes, f_scores, f_cids = [], [], []
    if len(idxs) > 0:
        for i in idxs.flatten():
            f_boxes.append(boxes[i])
            f_scores.append(scores[i])
            f_cids.append(cids[i])

    return f_boxes, f_scores, f_cids, infer_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--source", default="0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--skip", type=int, default=5, help="roda inferência a cada N frames")
    ap.add_argument("--show", action="store_true")
    # ordem real do seu modelo:
    # 0 crosswalk, 1 ped_light_go, 2 ped_light_stop, 3 ped_light_off
    ap.add_argument("--class-names", type=str,
                    default="crosswalk,ped_light_go,ped_light_stop,ped_light_off")
    ap.add_argument("--cooldown", type=float, default=2.0,
                    help="tempo mínimo entre duas falas iguais (s)")
    # áudios
    ap.add_argument("--audio-go", type=str, default="audio_go.mp3")
    ap.add_argument("--audio-stop", type=str, default="audio_stop.mp3")
    ap.add_argument("--audio-off", type=str, default="audio_off.mp3")
    ap.add_argument("--audio-cross", type=str, default="audio_crosswalk.mp3")
    ap.add_argument("--noaudio", action="store_true")
    # benchmark
    ap.add_argument("--no-benchmark", action="store_true",
                    help="desativa coleta e gráficos de benchmark")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="limite opcional de frames para benchmark (0 = sem limite)")
    args = ap.parse_args()

    class_names = [c.strip() for c in args.class_names.split(",")]
    print("[INFO] classes do modelo:", class_names)

    if not os.path.exists(args.onnx):
        print("[ERRO] modelo não encontrado:", args.onnx)
        return

    net = cv2.dnn.readNetFromONNX(args.onnx)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    src = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("[ERRO] não abriu fonte:", args.source)
        return

    history = deque(maxlen=10)
    last_spoken_time = 0.0

    # buffers pra NÃO piscar
    last_boxes, last_scores, last_cids, last_infer_ms = [], [], [], 0.0

    # ==========================
    # ARRAYS DO BENCHMARK
    # ==========================
    fps_list = []              # FPS instantâneo por frame (debug)
    frame_time_ms_list = []    # atraso total por frame (loop)
    infer_time_ms_list = []    # latência de inferência (quando roda a rede)

    # FPS REAL por segundo
    fps_per_second = []        # FPS calculado 1x/s
    frames_this_second = 0
    last_fps_time = time.time()
    current_fps_display = 0.0  # valor mostrado na tela

    # cria UMA janela só de vídeo (se show)
    if args.show:
        cv2.namedWindow("Semaforo+Faixa", cv2.WINDOW_NORMAL)

    frame_id = 0
    while True:
        t_loop_start = time.time()

        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        # limite opcional de frames (útil pra benchmark de vídeo gravado)
        if args.max_frames > 0 and frame_id > args.max_frames:
            print("[INFO] atingiu max_frames, encerrando loop.")
            break

        run_now = (frame_id % args.skip == 0)

        if run_now:
            boxes, scores, cids, infer_ms = run_inference(
                net, frame, args.imgsz, args.conf, args.iou
            )
            # atualiza buffer
            last_boxes, last_scores, last_cids, last_infer_ms = boxes, scores, cids, infer_ms

            if not args.no_benchmark:
                infer_time_ms_list.append(infer_ms)
        else:
            boxes, scores, cids, infer_ms = last_boxes, last_scores, last_cids, last_infer_ms

        # ver classes vistas
        seen = set()
        for cid in cids:
            if 0 <= cid < len(class_names):
                seen.add(class_names[cid])

        # prioridade
        if "crosswalk" in seen:
            current_state = "crosswalk"
        elif "ped_light_stop" in seen:
            current_state = "ped_light_stop"
        elif "ped_light_go" in seen:
            current_state = "ped_light_go"
        elif "ped_light_off" in seen:
            current_state = "ped_light_off"
        else:
            current_state = "none"

        history.append(current_state)
        # estado estável
        if history:
            counts = {}
            for s in history:
                counts[s] = counts.get(s, 0) + 1
            stable_state = max(counts, key=counts.get)
        else:
            stable_state = "none"

        now = time.time()
        if stable_state != "none" and (now - last_spoken_time) > args.cooldown:
            if stable_state == "ped_light_go":
                play_audio(args.audio_go, "semáforo aberto", mute=args.noaudio)
            elif stable_state == "ped_light_stop":
                play_audio(args.audio_stop, "aguarde, sinal vermelho", mute=args.noaudio)
            elif stable_state == "ped_light_off":
                play_audio(args.audio_off, "semáforo não identificado", mute=args.noaudio)
            elif stable_state == "crosswalk":
                play_audio(args.audio_cross, "faixa de pedestre detectada", mute=args.noaudio)
            last_spoken_time = now

        # fim do frame: calcula tempo total
        t_loop_end = time.time()
        dt = t_loop_end - t_loop_start

        if not args.no_benchmark:
            # FPS REAL por segundo
            frames_this_second += 1
            now_fps = time.time()
            if now_fps - last_fps_time >= 1.0:
                real_fps = frames_this_second
                fps_per_second.append(real_fps)
                current_fps_display = float(real_fps)  # valor mostrado no overlay
                frames_this_second = 0
                last_fps_time = now_fps

            # FPS instantâneo por frame (debug)
            if dt > 0:
                fps_inst = 1.0 / dt
                fps_list.append(fps_inst)
                frame_time_ms_list.append(dt * 1000.0)

        if args.show:
            # desenha resultados
            for (box, score, cid) in zip(boxes, scores, cids):
                x1, y1, x2, y2 = box
                name = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
                if name == "ped_light_go":
                    color = (0, 255, 0)
                elif name == "ped_light_stop":
                    color = (0, 0, 255)
                elif name == "crosswalk":
                    color = (255, 255, 0)
                else:
                    color = (255, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{name} {score:.2f}", (x1, max(15, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # texto do estado estável
            cv2.putText(frame, f"stable: {stable_state}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            # texto do FPS real na tela (canto superior direito)
            fps_text = f"FPS: {current_fps_display:.1f}"
            h, w = frame.shape[:2]
            cv2.putText(frame, fps_text, (w - 180, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            cv2.imshow("Semaforo+Faixa", frame)
            if cv2.waitKey(1) == 27:  # ESC
                break

    cap.release()
    if args.show:
        cv2.destroyAllWindows()
    print("[INFO] finalizado.")

    # ==========================
    # RESUMO E GRÁFICOS BENCHMARK
    # ==========================
    if not args.no_benchmark and fps_list and frame_time_ms_list:
        def stats(arr):
            return (sum(arr) / len(arr), min(arr), max(arr))

        # stats FPS por frame (instantâneo)
        avg_fps_frame, min_fps_frame, max_fps_frame = stats(fps_list)
        avg_frame_ms, min_frame_ms, max_frame_ms = stats(frame_time_ms_list)

        # stats FPS real por segundo
        if fps_per_second:
            avg_fps_real, min_fps_real, max_fps_real = stats(fps_per_second)
        else:
            avg_fps_real = min_fps_real = max_fps_real = 0.0

        # stats inferência
        if infer_time_ms_list:
            avg_infer_ms, min_infer_ms, max_infer_ms = stats(infer_time_ms_list)
        else:
            avg_infer_ms = min_infer_ms = max_infer_ms = 0.0

        print("\n========== BENCHMARK ==========")
        print(f"Frames analisados (loop):     {len(fps_list)}")
        print(f"Segundos medidos (FPS real):  {len(fps_per_second)}")
        print("--- FPS REAL (por segundo) ---")
        print(f"FPS real médio:               {avg_fps_real:.2f}")
        print(f"FPS real mínimo:              {min_fps_real:.2f}")
        print(f"FPS real máximo:              {max_fps_real:.2f}")
        print("--- FPS instantâneo (por frame, debug) ---")
        print(f"FPS inst. médio:              {avg_fps_frame:.2f}")
        print(f"FPS inst. mínimo:             {min_fps_frame:.2f}")
        print(f"FPS inst. máximo:             {max_fps_frame:.2f}")
        print("--- Atraso total do frame ---")
        print(f"Atraso total médio (ms):      {avg_frame_ms:.2f}")
        print(f"Atraso total mín (ms):        {min_frame_ms:.2f}")
        print(f"Atraso total máx (ms):        {max_frame_ms:.2f}")
        print("--- Latência de inferência ---")
        print(f"Latência inferência média:    {avg_infer_ms:.2f} ms")
        print(f"Latência inferência mín:      {min_infer_ms:.2f} ms")
        print(f"Latência inferência máx:      {max_infer_ms:.2f} ms")
        print("================================\n")

        # garante modo não interativo
        plt.ioff()

        # FPS REAL por segundo
        if fps_per_second:
            plt.figure()
            plt.plot(fps_per_second)
            plt.title("FPS real por segundo")
            plt.xlabel("Segundo")
            plt.ylabel("FPS")
            plt.savefig("benchmark_fps_real.png")
            plt.close()

        # FPS instantâneo por frame
        plt.figure()
        plt.plot(fps_list)
        plt.axhline(avg_fps_frame, linestyle="--")
        plt.title("FPS por frame (instantâneo)")
        plt.xlabel("Frame")
        plt.ylabel("FPS")
        plt.savefig("benchmark_fps.png")
        plt.close()

        # atraso total
        plt.figure()
        plt.plot(frame_time_ms_list)
        plt.axhline(avg_frame_ms, linestyle="--")
        plt.title("Atraso total por frame (ms)")
        plt.xlabel("Frame")
        plt.ylabel("Tempo (ms)")
        plt.savefig("benchmark_frame_time_ms.png")
        plt.close()

        # latência inferência
        if infer_time_ms_list:
            plt.figure()
            plt.plot(infer_time_ms_list)
            plt.axhline(avg_infer_ms, linestyle="--")
            plt.title("Latência de inferência (ms)")
            plt.xlabel("Inferências (frames com rede)")
            plt.ylabel("Tempo (ms)")
            plt.savefig("benchmark_infer_time_ms.png")
            plt.close()


if __name__ == "__main__":
    main()
