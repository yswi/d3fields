import matplotlib
# matplotlib.use('Agg')

from d3fields.utils.my_utils import depth2fgpcd, np2o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import cm
import open3d as o3d

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def newline(p1, p2):
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if p2[0] == p1[0]:
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    l = mlines.Line2D([xmin,xmax], [ymin,ymax])
    ax.add_line(l)
    return l

def draw_correspondence(img0, img1, kps0, kps1, matches=None, colors=None, max_draw_line_num=None, kps_color=(0,0,255),vert=False):
    if len(img0.shape)==2:
        img0=np.repeat(img0[:,:,None],3,2)
    if len(img1.shape)==2:
        img1=np.repeat(img1[:,:,None],3,2)

    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    if matches is None:
        assert(kps0.shape[0]==kps1.shape[0])
        matches=np.repeat(np.arange(kps0.shape[0])[:,None],2,1)

    if vert:
        w = max(w0, w1)
        h = h0 + h1
        out_img = np.zeros([h, w, 3], np.uint8)
        out_img[:h0, :w0] = img0
        out_img[h0:, :w1] = img1
    else:
        h = max(h0, h1)
        w = w0 + w1
        out_img = np.zeros([h, w, 3], np.uint8)
        out_img[:h0, :w0] = img0
        out_img[:h1, w0:] = img1

    for pt in kps0:
        pt = np.round(pt).astype(np.int32)
        cv2.circle(out_img, tuple(pt), 1, kps_color, -1)

    for pt in kps1:
        pt = np.round(pt).astype(np.int32)
        pt = pt.copy()
        if vert:
            pt[1] += h0
        else:
            pt[0] += w0
        cv2.circle(out_img, tuple(pt), 1, kps_color, -1)

    if max_draw_line_num is not None and matches.shape[0]>max_draw_line_num:
        np.random.seed(6033)
        idxs=np.arange(matches.shape[0])
        np.random.shuffle(idxs)
        idxs=idxs[:max_draw_line_num]
        matches= matches[idxs]

        if colors is not None and (type(colors)==list or type(colors)==np.ndarray):
            colors=np.asarray(colors)
            colors= colors[idxs]

    for mi,m in enumerate(matches):
        pt = np.round(kps0[m[0]]).astype(np.int32)
        pr_pt = np.round(kps1[m[1]]).astype(np.int32)
        if vert:
            pr_pt[1] += h0
        else:
            pr_pt[0] += w0
        if colors is None:
            cv2.line(out_img, tuple(pt), tuple(pr_pt), (0, 255, 0), 1)
        elif type(colors)==list or type(colors)==np.ndarray:
            color=(int(c) for c in colors[mi])
            cv2.line(out_img, tuple(pt), tuple(pr_pt), tuple(color), 1)
        else:
            color=(int(c) for c in colors)
            cv2.line(out_img, tuple(pt), tuple(pr_pt), tuple(color), 1)

    return out_img

def draw_keypoints(img,kps,colors=None,radius=2,thickness=-1):
    out_img=img.copy()
    for pi, pt in enumerate(kps):
        pt = np.round(pt).astype(np.int32)
        if colors is not None:
            color=[int(c) for c in colors[pi]]
            cv2.circle(out_img, tuple(pt), radius, color, thickness, cv2.FILLED)
        else:
            cv2.circle(out_img, tuple(pt), radius, (0,255,0), thickness)
    return out_img

def draw_epipolar_line(F, img0, img1, pt0, color):
    h1,w1=img1.shape[:2]
    hpt = np.asarray([pt0[0], pt0[1], 1], dtype=np.float32)[:, None]
    l = F @ hpt
    l = l[:, 0]
    a, b, c = l[0], l[1], l[2]
    pt1 = np.asarray([0, -c / b]).astype(np.int32)
    pt2 = np.asarray([w1, (-a * w1 - c) / b]).astype(np.int32)

    img0 = cv2.circle(img0, tuple(pt0.astype(np.int32)), 5, color, 2)
    img1 = cv2.line(img1, tuple(pt1), tuple(pt2), color, 2)
    return img0, img1

def draw_epipolar_lines(F, img0, img1,num=20):
    img0,img1=img0.copy(),img1.copy()
    h0, w0, _ = img0.shape
    h1, w1, _ = img1.shape

    for k in range(num):
        color = np.random.randint(0, 255, [3], dtype=np.int32)
        color = [int(c) for c in color]
        pt = np.random.uniform(0, 1, 2)
        pt[0] *= w0
        pt[1] *= h0
        pt = pt.astype(np.int32)
        img0, img1 = draw_epipolar_line(F, img0, img1, pt, color)

    return img0, img1

def gen_color_map(error, clip_max=12.0, clip_min=2.0, cmap_name='viridis'):
    rectified_error=(error-clip_min)/(clip_max-clip_min)
    rectified_error[rectified_error<0]=0
    rectified_error[rectified_error>=1.0]=1.0
    viridis=cm.get_cmap(cmap_name,256)
    colors=[viridis(e) for e in rectified_error]
    return np.asarray(np.asarray(colors)[:,:3]*255,np.uint8)

def scale_float_image(image):
    max_val, min_val = np.max(image), np.min(image)
    image = (image - min_val) / (max_val - min_val) * 255
    return image.astype(np.uint8)

def concat_images(img0,img1,vert=False):
    if not vert:
        h0,h1=img0.shape[0],img1.shape[0],
        if h0<h1: img0=cv2.copyMakeBorder(img0,0,h1-h0,0,0,borderType=cv2.BORDER_CONSTANT,value=0)
        if h1<h0: img1=cv2.copyMakeBorder(img1,0,h0-h1,0,0,borderType=cv2.BORDER_CONSTANT,value=0)
        img = np.concatenate([img0, img1], axis=1)
    else:
        w0,w1=img0.shape[1],img1.shape[1]
        if w0<w1: img0=cv2.copyMakeBorder(img0,0,0,0,w1-w0,borderType=cv2.BORDER_CONSTANT,value=0)
        if w1<w0: img1=cv2.copyMakeBorder(img1,0,0,0,w0-w1,borderType=cv2.BORDER_CONSTANT,value=0)
        img = np.concatenate([img0, img1], axis=0)

    return img


def concat_images_list(*args,vert=False):
    if len(args)==1: return args[0]
    img_out=args[0]
    for img in args[1:]:
        img_out=concat_images(img_out,img,vert)
    return img_out


def get_colors_gt_pr(gt,pr=None):
    if pr is None:
        pr=np.ones_like(gt)
    colors=np.zeros([gt.shape[0],3],np.uint8)
    colors[gt & pr]=np.asarray([0,255,0])[None,:]     # tp
    colors[ (~gt) & pr]=np.asarray([255,0,0])[None,:] # fp
    colors[ gt & (~pr)]=np.asarray([0,0,255])[None,:] # fn
    return colors


def draw_hist(fn,vals,bins=100,hist_range=None,names=None):
    if type(vals)==list:
        val_num=len(vals)
        if hist_range is None:
            hist_range = (np.min(vals),np.max(vals))
        if names is None:
            names=[str(k) for k in range(val_num)]
        for k in range(val_num):
            plt.hist(vals[k], bins=bins, range=hist_range, alpha=0.5, label=names[k])
        plt.legend()
    else:
        if hist_range is None:
            hist_range = (np.min(vals),np.max(vals))
        plt.hist(vals,bins=bins,range=hist_range)

    plt.savefig(fn)
    plt.close()

def draw_pr_curve(fn,gt_sort):
    pos_num_all=np.sum(gt_sort)
    pos_nums=np.cumsum(gt_sort)
    sample_nums=np.arange(gt_sort.shape[0])+1
    precisions=pos_nums.astype(np.float64)/sample_nums
    recalls=pos_nums/pos_num_all

    precisions=precisions[np.arange(0,gt_sort.shape[0],gt_sort.shape[0]//40)]
    recalls=recalls[np.arange(0,gt_sort.shape[0],gt_sort.shape[0]//40)]
    plt.plot(recalls,precisions,'r-')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.savefig(fn)
    plt.close()

def draw_features_distribution(fn,feats,colors,ds_type='pca'):
    n,d=feats.shape
    if d>2:
        if ds_type=='pca':
            pca=PCA(2)
            feats=pca.fit_transform(feats)
        elif ds_type=='tsne':
            tsne=TSNE(2)
            feats=tsne.fit_transform(feats)
        elif ds_type=='pca-tsne':
            if d>50:
                tsne=PCA(50)
                feats=tsne.fit_transform(feats)
            tsne=TSNE(2,100.0)
            feats=tsne.fit_transform(feats)
        else:
            raise NotImplementedError

    colors=[np.array([c[0],c[1],c[2]],np.float64)/255.0 for c in colors]
    feats_min=np.min(feats,0,keepdims=True)
    feats_max=np.max(feats,0,keepdims=True)
    feats=(feats-(feats_min+feats_max)/2)*10/(feats_max-feats_min)
    plt.scatter(feats[:,0],feats[:,1],s=0.5,c=colors)
    plt.savefig(fn)
    plt.close()
    return feats

def draw_points(img,points):
    pts=np.round(points).astype(np.int32)
    h,w,_=img.shape
    pts[:,0]=np.clip(pts[:,0],a_min=0,a_max=w-1)
    pts[:,1]=np.clip(pts[:,1],a_min=0,a_max=h-1)
    img=img.copy()
    img[pts[:,1],pts[:,0]]=255
    # img[pts[:,1],pts[:,0]]+=np.asarray([127,0,0],np.uint8)[None,:]
    return img

def draw_bbox(img,bbox,color=None):
    if color is not None:
        color=[int(c) for c in color]
    else:
        color=(0,255,0)
    img=cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),color)
    return img

def output_points(fn,pts,colors=None):
    with open(fn, 'w') as f:
        for pi, pt in enumerate(pts):
            f.write(f'{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} ')
            if colors is not None:
                f.write(f'{int(colors[pi,0])} {int(colors[pi,1])} {int(colors[pi,2])}')
            f.write('\n')

def compute_axis_points(pose):
    R=pose[:,:3] # 3,3
    t=pose[:,3:] # 3,1
    pts = np.concatenate([np.identity(3),np.zeros([3,1])],1) # 3,4
    pts = R.T @ (pts - t)
    colors = np.asarray([[255,0,0],[0,255,0,],[0,0,255],[0,0,0]],np.uint8)
    return pts.T, colors

def aggr_point_cloud(database, sel_time=None, downsample=True):
    img_ids = database.get_img_ids()
    start = 0
    end = len(img_ids)
    # end = 2
    step = 1
    # visualize scaled COLMAP poses
    COLMAP_geometries = []
    base_COLMAP_pose = database.get_base_pose()
    base_COLMAP_pose = np.concatenate([base_COLMAP_pose, np.array([[0, 0, 0, 1]])], axis=0)
    for i in range(start, end, step):
        id = img_ids[i]
        depth = database.get_depth(id, sel_time=sel_time)
        color = database.get_image(id, sel_time=sel_time) / 255.
        K = database.get_K(id)
        cam_param = [K[0,0], K[1,1], K[0,2], K[1,2]] # fx, fy, cx, cy
        mask = database.get_mask(id, sel_time=sel_time) & (depth > 0)
        pcd = depth2fgpcd(depth, mask, cam_param)
        pose = database.get_pose(id)
        pose = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)
        pose = base_COLMAP_pose @ np.linalg.inv(pose)
        # print('COLMAP pose:')
        # print(pose)
        # pose[:3, 3] = pose[:3, 3] / 82.5
        trans_pcd = pose @ np.concatenate([pcd.T, np.ones((1, pcd.shape[0]))], axis=0)
        trans_pcd = trans_pcd[:3, :].T
        pcd_o3d = np2o3d(trans_pcd, color[mask])
        radius = 0.01
        # downsample
        if downsample:
            pcd_o3d = pcd_o3d.voxel_down_sample(radius)
        COLMAP_geometries.append(pcd_o3d)
    aggr_geometry = o3d.geometry.PointCloud()
    for pcd in COLMAP_geometries:
        aggr_geometry += pcd
    return aggr_geometry

def voxel_downsample(pcd, voxel_size, pcd_color=None):
    # :param pcd: [N,3] numpy array
    # :param voxel_size: float
    # :return: [M,3] numpy array
    pcd = np2o3d(pcd, pcd_color)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    if pcd_color is not None:
        return np.asarray(pcd_down.points), np.asarray(pcd_down.colors)
    else:
        return np.asarray(pcd_down.points)
def _create_uniform_pixel_coords_image(resolution: np.ndarray):
    pixel_x_coords = np.reshape(
        np.tile(np.arange(resolution[1]), [resolution[0]]),
        (resolution[0], resolution[1], 1),
    ).astype(np.float32)
    pixel_y_coords = np.reshape(
        np.tile(np.arange(resolution[0]), [resolution[1]]),
        (resolution[1], resolution[0], 1),
    ).astype(np.float32)
    pixel_y_coords = np.transpose(pixel_y_coords, (1, 0, 2))
    uniform_pixel_coords = np.concatenate(
        (pixel_x_coords, pixel_y_coords, np.ones_like(pixel_x_coords)), -1
    )
    return uniform_pixel_coords

def pointcloud_from_depth_and_camera_params(
    depth: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray
) -> np.ndarray:
    """Converts depth (in meters) to point cloud in word frame.
    :return: A numpy array of size (width, height, 3)
    """
    upc = _create_uniform_pixel_coords_image(depth.shape)
    pc = upc * np.expand_dims(depth, -1)
    C = np.expand_dims(extrinsics[:3, 3], 0).T
    R = extrinsics[:3, :3]
    R_inv = R.T  # inverse of rot matrix is transpose
    R_inv_C = np.matmul(R_inv, C)
    extrinsics = np.concatenate((R_inv, -R_inv_C), -1)
    cam_proj_mat = np.matmul(intrinsics, extrinsics)
    cam_proj_mat_homo = np.concatenate([cam_proj_mat, [np.array([0, 0, 0, 1])]])
    cam_proj_mat_inv = np.linalg.inv(cam_proj_mat_homo)[0:3]
    world_coords_homo = np.expand_dims(_pixel_to_world_coords(pc, cam_proj_mat_inv), 0)
    world_coords = world_coords_homo[..., :-1][0]
    return world_coords
    
def _pixel_to_world_coords(pixel_coords, cam_proj_mat_inv):
    h, w = pixel_coords.shape[:2]
    pixel_coords = np.concatenate([pixel_coords, np.ones((h, w, 1))], -1)
    world_coords = _transform(pixel_coords, cam_proj_mat_inv)
    world_coords_homo = np.concatenate([world_coords, np.ones((h, w, 1))], axis=-1)
    return world_coords_homo

def _transform(coords, trans):
    h, w = coords.shape[:2]
    coords = np.reshape(coords, (h * w, -1))
    coords = np.transpose(coords, (1, 0))
    transformed_coords_vector = np.matmul(trans, coords)
    transformed_coords_vector = np.transpose(transformed_coords_vector, (1, 0))
    return np.reshape(transformed_coords_vector, (h, w, -1))    

def aggr_point_cloud_from_data(colors, depths, Ks, poses, downsample=True, masks=None, boundaries=None, out_o3d=True):
    # colors: [N, H, W, 3] numpy array in uint8
    # depths: [N, H, W] numpy array in meters
    # Ks: [N, 3, 3] numpy array
    # poses: [N, 4, 4] numpy array
    # masks: [N, H, W] numpy array in bool
    N, H, W, _ = colors.shape
    colors = colors / 255.
    start = 0
    end = N
    # end = 2
    step = 1
    # visualize scaled COLMAP poses
    pcds = []
    pcd_colors = []
    for i in range(start, end, step):
        depth = depths[i]
        color = colors[i]
        K = Ks[i]
        cam_param = [K[0,0], K[1,1], K[0,2], K[1,2]] # fx, fy, cx, cy
        if masks is None:
            mask = (depth > 0) & (depth < 1.5)
        else:
            mask = masks[i] & (depth > 0)
        # mask = np.ones_like(depth, dtype=bool)

       

        pose = poses[i]
        trans_pcd = pointcloud_from_depth_and_camera_params(depth, pose, K).reshape(-1,3)
        trans_pcd = trans_pcd[mask.reshape(-1) == 1]
        # trans_pcd_ = depth2fgpcd(depth, mask, cam_param)



        import open3d as o3d
        
        # Create a point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(trans_pcd)

   
        # Save the point cloud to a .ply file
        o3d.io.write_point_cloud("my_pcd.ply", pcd)
        print(trans_pcd.shape)

        pose = np.linalg.inv(pose)
        
        # trans_pcd = pose @ np.concatenate([pcd.T, np.ones((1, pcd.shape[0]))], axis=0)
        # trans_pcd = trans_pcd[:3, :].T
        
        # plt.subplot(1, 4, 1)
        # plt.imshow(trans_pcd[:, 0].reshape(H, W))
        # plt.subplot(1, 4, 2)
        # plt.imshow(trans_pcd[:, 1].reshape(H, W))
        # plt.subplot(1, 4, 3)
        # plt.imshow(trans_pcd[:, 2].reshape(H, W))
        # plt.subplot(1, 4, 4)
        # plt.imshow(color)
        # plt.show()
        
        if boundaries is not None:
            x_lower = boundaries['x_lower']
            x_upper = boundaries['x_upper']
            y_lower = boundaries['y_lower']
            y_upper = boundaries['y_upper']
            z_lower = boundaries['z_lower']
            z_upper = boundaries['z_upper']

            # print(np.amin(trans_pcd, axis= 0)  , np.amax(trans_pcd, axis= 0) )
            
            trans_pcd_mask = (trans_pcd[:, 0] > x_lower) &\
                (trans_pcd[:, 0] < x_upper) &\
                    (trans_pcd[:, 1] > y_lower) &\
                        (trans_pcd[:, 1] < y_upper) &\
                            (trans_pcd[:, 2] > z_lower) &\
                                (trans_pcd[:, 2] < z_upper)
            
            if out_o3d:
                pcd_o3d = np2o3d(trans_pcd[trans_pcd_mask], color[mask][trans_pcd_mask])
            else:
                pcd_np = trans_pcd[trans_pcd_mask]
                pcd_color = color[mask][trans_pcd_mask]
        else:
            if out_o3d:
                pcd_o3d = np2o3d(trans_pcd, color[mask])
            else:
                pcd_np =  trans_pcd
                pcd_color = color[mask]
        # downsample
        if out_o3d:
            if downsample:
                radius = 0.01
                pcd_o3d = pcd_o3d.voxel_down_sample(radius)
            pcds.append(pcd_o3d)
        else:
            if downsample:
                pcd_np, pcd_color = voxel_downsample(pcd_np, 0.01, pcd_color)
            pcds.append(pcd_np)
            pcd_colors.append(pcd_color)
    if out_o3d:
        aggr_pcd = o3d.geometry.PointCloud()
        for pcd in pcds:
            aggr_pcd += pcd
        return aggr_pcd
    else:
        pcds = np.concatenate(pcds, axis=0)
        pcd_colors = np.concatenate(pcd_colors, axis=0)
        return pcds, pcd_colors

def draw_pcd_with_spheres(pcd, sphere_np_ls):
    # :param pcd: o3d.geometry.PointCloud
    # :param sphere_np_ls: list of [N, 3] numpy array
    sphere_list = []
    color_list = [
        [1.,0.,0.],
        [0.,1.,0.],
    ]
    for np_idx, sphere_np in enumerate(sphere_np_ls):
        for src_pt in sphere_np:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere.compute_vertex_normals()
            sphere.paint_uniform_color(color_list[np_idx])
            sphere.translate(src_pt)
            sphere_list.append(sphere)
    o3d.visualization.draw_geometries([pcd] + sphere_list)

# 360 visualization from jupyter notebook
def rotate_pcd(pcd, eye = [-1.25, 2, 0.5]):
    fig = go.Figure(
      data=pcd,
      layout=dict(
          scene=dict(
              xaxis=dict(visible=False),
              yaxis=dict(visible=False),
              zaxis=dict(visible=False)
          )
      )
    )

    x_eye = eye[0]
    y_eye = eye[1]
    z_eye = eye[2]

    fig.update_layout(
            title='360 animation',
            width=600,
            height=600,
            scene_camera_eye=dict(x=x_eye, y=y_eye, z=z_eye),
            updatemenus=[dict(type='buttons',
                    showactive=False,
                    y=1,
                    x=0.8,
                    xanchor='left',
                    yanchor='bottom',
                    pad=dict(t=45, r=10),
                    buttons=[dict(label='Play',
                                    method='animate',
                                    args=[None, dict(frame=dict(duration=5, redraw=True), 
                                                                transition=dict(duration=0),
                                                                fromcurrent=True,
                                                                mode='immediate'
                                                                )]
                                                )
                                        ]
                                )
                            ]
    )

  
    def rotate_z(x, y, z, theta):
        w = x+1j*y
        return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z
    
    frames=[]
    for t in np.arange(0, 6.26, 0.1):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(go.Frame(layout=dict(scene_camera_eye=dict(x=xe, y=ye, z=ze))))
    fig.frames=frames
    return frames
    # fig.show