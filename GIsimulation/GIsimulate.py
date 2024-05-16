from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
class GhostImaging:
    """
    params
    --------
    input_img_path: Path of img which you want to Imaging 
    img_width: width of img
    img_height: height of img
    """
    def __init__(self, input_img_path: str, img_width: int, img_height: int):
        self.width = img_width
        self.height = img_height

        # Img Processing
        pil_img = Image.open(input_img_path).convert('L')   # Open img file and convert to 8bit(0-255)
        pil_img = pil_img.resize((self.width, self.height)) # Resize img
        self.obj = np.array(pil_img, dtype=np.float64) / 255 # Convert to ndarray and normalize
        self.obj = np.where(self.obj > 0.5, 1, 0)           # white to 1, black to 0

        # Initialize results
        self.result_img = np.zeros((self.height, self.width))

        # Save evaluation
        self.evals = np.array([])

        # for animation
        self.fig = plt.figure()
        self.frames = []
    
    def simulate_gi(self, pattern_num: int, frame_num: int, use_average: bool = False):
        # Reconstruction obj
        recon_obj = np.zeros((self.height, self.width))
        if(use_average):
            intensities = np.array([])
            pattern_sum = np.zeros((self.height, self.width))
        
        for count in range(pattern_num):
            number_of_patterns = count + 1
            
            # Generate random pattern
            pattern = np.where(np.random.rand(self.height, self.width) > 0.5, 1, 0)

            # pattern * img_obj
            mul_mat = pattern * self.obj

            # Receive signals within ONE PD
            intensity = np.sum(np.abs(mul_mat)**2)

            # Reconstruct img from recieved intensity
            if (use_average):
                intensities = np.append(intensities, intensity)
                recon_obj, pattern_sum = self.reconstruction_with_avg(intensity, intensities,
                                                                      pattern, pattern_sum, recon_obj)
            else:
                recon_obj = self.reconstruction(intensity, pattern, recon_obj)
            
            # Evaluation with MSE
            mse = self.mean_squared_error()
            self.evals = np.append(self.evals, mse)

            if number_of_patterns % int(pattern_num / frame_num) == 0:
                print(f'number of patterns: {number_of_patterns}, MSE: {mse:.4f}')
                frame = plt.imshow(self.result_img, cmap='binary_r', vmin=0, vmax=1)
                text = plt.text(-0.4, 0.8, f'number of patterns = {number_of_patterns}')
                self.frames.append([frame, text])
    
    def reconstruction(self, intensity, pattern, recon_obj):
        recon_obj += pattern * intensity
        max_val = np.amax(recon_obj)
        min_val = np.amin(recon_obj)
        self.result_img = (recon_obj - min_val) / (max_val - min_val)
        return recon_obj

    def reconstruction_with_avg(self, intensity, intensities, pattern, pattern_sum, recon_obj):
        intensity_ave = np.mean(intensities)
        pattern_sum += pattern
        recon_obj += intensity * pattern
        recon_obj_sub_avg = recon_obj - intensity_ave * pattern_sum # ここで差っ引く
        max_val = np.amax(recon_obj_sub_avg) # 最大値取得
        min_val = np.amin(recon_obj_sub_avg) # 最小値取得
        if max_val != min_val:
            self.result_img = (recon_obj_sub_avg - min_val) / (max_val - min_val) # 1に正規化して結果として格納
        else:
            self.result_img = recon_obj_sub_avg
        return recon_obj, pattern_sum
    
    def mean_squared_error(self):
        return np.sum((self.obj - self.result_img)**2) / (self.height * self.width)
    
    # 撮像結果表示
    def show_results(self, save_path: str):
        # アニメーション保存
        ani = ArtistAnimation(self.fig, self.frames, interval=10)
        ani.save(save_path + "result_ani.gif", writer='pillow', fps=10)
        plt.close()
        # 画像表示
        plt.subplots()
        plt.imshow(self.obj, cmap='binary_r', vmin=0, vmax=1)
        plt.title('target object')
        plt.colorbar()
        plt.subplots()
        plt.imshow(self.result_img, cmap='binary_r', vmin=0, vmax=1)
        plt.title('reconstructed result')
        plt.colorbar()
        plt.subplots()
        plt.plot(np.arange(len(self.evals)) + 1, self.evals)
        plt.xlabel('number of patterns')
        plt.ylabel('MSE')
        plt.ylim(0, 0.5)
        plt.grid()
        plt.show()

def main():
    '''
    (1) 初期処理
    input_img_path: 撮像対象として入力する画像ファイルのパス
    img_width     : 撮像対象の幅（この大きさにリサイズする）
    img_height    : 撮像対象の高さ（この大きさにリサイズする）
    '''
    gi = GhostImaging(input_img_path = 'GIsimulation/GI_16x16_white_on_black.png', img_width = 16, img_height = 16)

    '''
    (2) ゴーストイメージング開始
    pattern_num: 使うパターンの枚数
    frame_num  : gif化する際のフレームの総数(> 1)
    use_average: 再構成時に全強度の平均値を使うか？(デフォルトはFalseで使わない)
    '''
    gi.simulate_gi(pattern_num = 100000, frame_num = 50, use_average = True)

    '''
    (3) 結果表示
    save_path: gifを保存するパス
    '''
    gi.show_results(save_path = "GIsimulation/")

if __name__ == "__main__":
    main()