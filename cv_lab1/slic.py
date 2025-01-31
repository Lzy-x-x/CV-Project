import os
import numpy as np
import cv2
from tqdm import tqdm

class SLIC:
    def __init__(self, img: np.ndarray, step: int, nc: int):
        self.img = img
        self.ht, self.wd = img.shape[:2]
        self.labimg = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB).astype(np.float64)
        self.step = step
        self.nc = nc  # Color proximity weight
        self.ns = step  # Spatial proximity weight
        self.max_iter = 20  # Maximum number of iterations
        
    def _in(self, x: int, y: int) -> bool:
        """
        Check if the given coordinates are inside the image boundaries.
        """
        return 0 <= x < self.ht and 0 <= y < self.wd
    
    def _find_locmin(self, center):
        """
        Find the local minimum gradient position in a 3x3 neighborhood around the center.
        
        Parameters:
            center (tuple): The (x, y) coordinates of the center.
            
        Returns:
            tuple: The (x, y) coordinates of the adjusted center with minimum gradient.
        """
        min_grad = np.inf
        nc = center
        if not self._in(nc[0] + 2, nc[1] + 2) or not self._in(nc[0] - 1, nc[1] - 1):
            return nc
        for x in range(center[0] - 1, center[0] + 2):
            for y in range(center[1] - 1, center[1] + 2):
                c = self.labimg[x, y]
                c1 = self.labimg[x + 1, y]
                c2 = self.labimg[x, y + 1]
                grad = abs(c1[0] - c[0]) + abs(c2[0] - c[0])
                if grad < min_grad:
                    min_grad = grad
                    nc = (x, y)
        return nc
    
    def _init_data(self):
        """
        Initialize the labels and cluster centers.
        """
        # x: top->bottom, y: left->right
        self.labels = np.full((self.ht, self.wd), -1, dtype=np.int32)
        
        centers = []
        for x in range(self.step, self.ht - self.step//2, self.step):
            for y in range(self.step, self.wd - self.step//2, self.step):
                nc = self._find_locmin((x, y))
                color = self.labimg[nc]
                center = (color[0], color[1], color[2], nc[0], nc[1])
                centers.append(center)
        self.centers = np.array(centers)
        
    def cluster(self):
        """
        Perform the clustering to segment the image into superpixels.
        """
        self._init_data()
        
        for i in range(self.max_iter):
            self.dists = np.full((self.ht, self.wd), np.inf)
            K = len(self.centers)
            
            for k in range(K):
                center = self.centers[k]
                x, y = int(center[3]), int(center[4])
                xl, xr = max(0, x - self.step), min(self.ht, x + self.step)
                yl, yr = max(0, y - self.step), min(self.wd, y + self.step)
                crop = self.labimg[xl:xr, yl:yr]
                
                col_dist = np.linalg.norm(crop - center[:3], axis=2)
                xx, yy = np.ogrid[xl:xr, yl:yr]
                spa_dist = np.sqrt((xx-x)**2 + (yy-y)**2)
                dist = np.sqrt((col_dist/self.nc)**2 + (spa_dist/self.ns)**2)
                
                dist_crop = self.dists[xl:xr, yl:yr]
                idx = dist < dist_crop
                dist_crop[idx] = dist[idx]
                self.labels[xl:xr, yl:yr][idx] = k
                
            for k in range(K):
                idx = self.labels == k
                col = self.labimg[idx]
                dist = np.array(np.where(idx)).T
                
                self.centers[k][:3] = np.sum(col, axis=0)
                self.centers[k][3:] = np.sum(dist, axis=0)
                self.centers[k] /= np.sum(idx)
                
    def merge(self):
        """
        Enforce connectivity by merging small isolated regions.
        """
        dx4 = [1, 0, -1, 0]
        dy4 = [0, 1, 0, -1]
        label = 0
        adjlabel = 0
        thr = self.ht * self.wd // len(self.centers) // 4
        new_labels = np.full((self.ht, self.wd), -1, dtype=np.int32)
        pixels = []
        
        for x in range(self.ht):
            for y in range(self.wd):
                if new_labels[x, y] == -1:
                    pixels = [(x, y)]
                    for dx, dy in zip(dx4, dy4):
                        nx, ny = x + dx, y + dy
                        if self._in(nx, ny) and new_labels[nx, ny] >= 0:
                            adjlabel = new_labels[nx, ny]
                    cnt = 1
                    cur = 0
                    while cur < cnt:
                        for dx, dy in zip(dx4, dy4):
                            nx, ny = pixels[cur][0] + dx, pixels[cur][1] + dy
                            if self._in(nx, ny) and new_labels[nx, ny] == -1 and self.labels[nx, ny] == self.labels[x, y]:
                                pixels.append((nx, ny))
                                new_labels[nx, ny] = label
                                cnt += 1
                        cur += 1
                    if cnt <= thr:
                        for p in pixels:
                            new_labels[p] = adjlabel
                        label -= 1
                    label += 1
        self.labels = new_labels
        self.np = label + 1
                
    def contour(self, color=(255, 0, 0)):
        """
        Draw the superpixel contours on the image.
        
        Parameters:
            color (tuple): BGR color to draw the contours.
        """
        dx8 = [1, 1, 0, -1, -1, -1, 0, 1]
        dy8 = [0, 1, 1, 1, 0, -1, -1, -1]
        is_edge = np.zeros((self.ht, self.wd), dtype=np.bool)
        
        for x in range(self.ht):
            for y in range(self.wd):
                c = 0
                for dx, dy in zip(dx8, dy8):
                    nx, ny = x + dx, y + dy
                    if self._in(nx, ny) and self.labels[nx, ny] != self.labels[x, y]:
                        c += 1
                if c >= 2:
                    is_edge[x, y] = True
        
        self.img[is_edge] = color
        
    def color(self, p=0.2):
        """
        Overlay random colors onto the superpixels.
        
        Parameters:
            p (float): Blending factor between the original image and the superpixel colors.
        """
        colors = np.random.randint(0, 255, (self.np, 3))
        self.img = (1-p)*self.img + p*colors[self.labels]
    
def main(n, nc):
    """
    Main function to process images and apply SLIC superpixel segmentation.
    
    - Reads images from the data directory.
    - Applies SLIC to each image with the specified parameters.
    - Saves the processed images to the output directory.
    
    Parameters:
        n (int): Desired number of superpixels.
        nc (int): Compactness factor for color proximity.
    """
    data_dir = "data"
    output_dir = f"output/{n}_{nc}"
    
    print(f"Processing n={n}, nc={nc}")
    
    if os.path.exists(output_dir):
        print("Output directory already exists")
        return
    os.makedirs(output_dir)
    files = os.listdir(data_dir)
    
    for img_name in tqdm(files):
        img = cv2.imread(os.path.join(data_dir, img_name))
        step = int(np.sqrt(img.shape[0]*img.shape[1]/n))
        
        slic = SLIC(img, step, nc)
        slic.cluster()
        slic.merge()
        # slic.color()
        slic.contour()
        cv2.imwrite(os.path.join(output_dir, img_name), slic.img)
        
if __name__ == "__main__":
    for n in [100, 400, 900]:
        for nc in [10, 20, 40, 80, 160]:
            main(n, nc)