# Smart Maze (Q-Learning)

Smart Maze adalah game grid 2D tempat agen AI belajar menemukan jalur optimal dengan menghindari tembok dan jebakan menggunakan Q-Learning dan kebijakan ε-greedy.

## Fitur
- Maze berbasis JSON (mudah diganti).
- Visualisasi grafis dengan Pygame.
- Grafik performa pelatihan (Matplotlib).
- Struktur kode terpisah dan bersih (clean code).

## Persamaan Pembaruan Q

$$Q(s,a) \leftarrow Q(s,a) + \alpha \big[r + \gamma \max_{a'} Q(s', a') - Q(s,a)\big]$$

## Struktur
- `src/maze_env.py`: Environment maze (state, action, reward, step, reset).
- `src/q_learning_agent.py`: Agen Q-learning (Q-table, ε-greedy, update).
- `src/config_loader.py`: Loader & validator konfigurasi JSON.
- `src/visualizer.py`: Visualisasi Pygame.
- `src/trainer.py`: Pelatihan, logging metrik, plotting.
- `src/game.py`: Entry point untuk menjalankan game/training.
- `configs/`: Kumpulan file JSON maze.

## Instalasi Dependensi

Pastikan virtual environment aktif, lalu:

```bash
pip install -r smart_maze/requirements.txt
```

## Cara Menjalankan

- Pelatihan tanpa visualisasi (lebih cepat):
```bash
python -m smart_maze.src.trainer --config smart_maze/configs/maze_10x10.json --episodes 500
```

- Dengan visualisasi game:
```bash
python -m smart_maze.src.game --config smart_maze/configs/maze_10x10.json --episodes 300 --render --cell-size 32
```

- Plot metrik dari pelatihan disimpan otomatis ke `smart_maze/assets/metrics.png`. Tambahkan `--show-plot` untuk menampilkan.

## Konfigurasi JSON
Contoh format:
```json
{
  "width": 10,
  "height": 10,
  "start": [0, 0],
  "goal": [9, 9],
  "walls": [[1,0], [1,1]],
  "traps": [[3,3], [5,6]],
  "rewards": {
    "goal": 100,
    "trap": -30,
    "wall": -10,
    "step": -1
  }
}
```

## Catatan
- Untuk Windows, jika jendela Pygame tidak muncul, pastikan tidak ada proses Python lain yang memblok.
- Nilai hyperparameter (`episodes`, `epsilon`, `alpha`, dll.) dapat diatur via argumen CLI.
