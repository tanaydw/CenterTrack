import cv2


def get_u(x, long_org, scale_u):
    return int((x - long_org) / scale_u)


def get_v(x, lat_org, scale_v):
    return int((x - lat_org) / scale_v)


class map(object):

    def __init__(self, 
                sat_map='satmap/negley_map.png',
                p1 = (40.46984126496397, -79.93069034546637),
                p2 = (40.46695644291853, -79.92597035690083),
                csv_num=3):    
        
        # Read Satellite Metadata
        self.sat_img = cv2.cvtColor(cv2.imread(sat_map), cv2.COLOR_BGR2RGB)
        self.segmented_image = self.get_segmented_map()

        # MANUAL
        self.p1 = p1
        self.p2 = p2
        self.lat_org, self.long_org, self.scale_u, self.scale_v = self.get_map_vectors()

        # Reading CSV files
        csv_list = []
        for i in range(csv_num):
            csv_list.append(pd.read_csv(opt.base_dir + '1_' + str(i+1) + '_all.csv', 
            encoding = "ISO-8859-1")[['latitude', 'longitude']])
        self.csv = pd.concat(csv_list)
        self.csv.reset_index(drop=True, inplace=True)

        # Transformation
        self.csvt = csv.copy()
        self.csvt['longitude'] = self.csvt['longitude'].apply(lambda x: get_u(x, long_org, scale_u))
        self.csvt['latitude'] = self.csvt['latitude'].apply(lambda x: get_v(x, lat_org, scale_v))


    def get_segmented_map(self):
        img = self.sat_img
        segmented_image = np.zeros(img.shape, dtype=np.uint8)

        # Defining background color = (254, 204, 165)
        segmented_image[:, :, 0] = 254
        segmented_image[:, :, 1] = 204
        segmented_image[:, :, 2] = 165

        # Segmenting Road (255, 255, 255)
        road_mask = (img[:, :, 0] == 255) & (img[:, :, 1] == 255) & (img[:, :, 2] == 255)
        segmented_image[road_mask, 0] = 219
        segmented_image[road_mask, 1] = 108
        segmented_image[road_mask, 2] = 115

        # Segmenting Buildings (241, 241, 241)
        build_mask = (img[:, :, 0] == 241) & (img[:, :, 1] == 241) & (img[:, :, 2] == 241)
        
        """n_opt = 3
        for _ in range(n_opt):
            build_mask = morphology.binary_erosion(build_mask)
        for _ in range(n_opt):
            build_mask = morphology.binary_dilation(build_mask)"""
        
        segmented_image[build_mask, 0] = 254
        segmented_image[build_mask, 1] = 137
        segmented_image[build_mask, 2] = 9
        
        return segmented_image

    def get_map_vectors(self):
        u1, v1 = 0, 0
        u2, v2 = self.segmented_image.shape[1], self.segmented_image.shape[0]
        lat1, long1 = self.p1
        lat2, long2 = self.p2

        scale_u = (long1 - long2) / (u1 - u2)
        scale_v = (lat1 - lat2) / (v1 - v2)
        lat_org = lat1 - scale_v * v1
        long_org = long1 - scale_u * u1
        
        return lat_org, long_org, scale_u, scale_v


# CHANGE: Added 'if' statement for Bird's Eye Transformation
if opt.bev:
    # Getting Slope
    scale = 30
    dst_x, dst_z = 1000, 2500
    lon = csv.loc[cnt-1:cnt-1+opt.fpts, 'longitude'].to_numpy()
    lat = csv.loc[cnt-1:cnt-1+opt.fpts, 'latitude'].to_numpy()
    theta = 180 + slope_calculator(lon, lat)
    
    # Getting BEV Image
    latitude = csvt.loc[cnt-1, 'latitude']
    longitude = csvt.loc[cnt-1, 'longitude']
    birdimage = get_patch(segmented_image, latitude, longitude, theta, dst_x, dst_z)
    img = get_bev(results[cnt], opt, scale, birdimage)

    # Writing to Video
    rows_rgb, cols_rgb, channels = ret['generic'].shape
    img = image_resize(img, height = rows_rgb)
    rows_gray, cols_gray, _ = img.shape
    rows_comb = max(rows_rgb, rows_gray)
    cols_comb = cols_rgb + cols_gray
    comb = np.zeros(shape=(rows_comb, cols_comb, channels), dtype=np.uint8)
    comb[:rows_rgb, :cols_rgb] = ret['generic']
    comb[:rows_gray, cols_rgb:] = img

    if not out2_init:
        vw, vh = comb.shape[0], comb.shape[1]
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out2 = cv2.VideoWriter('./results/{}_bev.avi'.format(
            opt.exp_id + '_' + out_name), fourcc, opt.save_framerate, (vh, vw))
        out2_init = True
    
    comb = cv2.resize(comb, (vh, vw))
    out2.write(comb)