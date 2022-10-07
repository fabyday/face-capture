
import dlib

from libkinect2 import Kinect2
from libkinect2.utils import depth_map_to_image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import igl 

from qpsolvers import solve_qp


v, _ = igl.read_triangle_mesh("generic_neutral_mesh.obj")
s = []
with open("face_landmark.txt", 'r') as fp:
    l = None
    while True :
        l = fp.readline()
        if not l :
            break
        s.append(int(l))

print(s)
v_all = v 
v = v[s]
print(v)
a = v[:, :2].T
blendshapes = []
import glob 
for i in glob.glob("shapes/**.obj"):
    vf, _ = igl.read_triangle_mesh(i)
    blendshapes.append(vf)
blendshapes = np.array(blendshapes)
# blendshapes -= v_all[None, ...] #for testing
full_blendshape = blendshapes - v_all [None, ...]
blendshapes = blendshapes[:, s, :]
print(blendshapes.shape)
w0 = np.zeros(full_blendshape.shape[0])

# plt.scatter(a[0], a[1], c = np.arange(len(a[0])))

# plt.show()



# Init Kinect2 w/2 sensors
kinect = Kinect2(use_sensors=['color', 'depth'])
kinect.connect()
kinect.wait_for_worker()



# create list for landmarks
ALL = list(range(0, 68))
RIGHT_EYEBROW = list(range(17, 22))
LEFT_EYEBROW = list(range(22, 27))
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
NOSE = list(range(27, 36))
MOUTH_OUTLINE = list(range(48, 61))
MOUTH_INNER = list(range(61, 68))
JAWLINE = list(range(0, 17))


# create face detector, predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# cap = cv2.VideoCapture("udp://127.0.0.1:10000")




for _, color_img, depth_map in kinect.iter_frames():
    
    # image = cv2.resize(color_img, dsize=(640, 480), interpolation=cv2.INTER_AREA)
    image = color_img
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get faces (up-sampling=1)
    face_detector = detector(img_gray, 1)
    # the number of face detected
    print("The number of faces detected : {}".format(len(face_detector)))

    # loop as the number of face
    # one loop belong to one face
    for face in face_detector:
        # face wrapped with rectangle
        cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()),
                        (0, 0, 255), 3)

        # make prediction and transform to numpy array
        landmarks = predictor(image, face)  # 얼굴에서 68개 점 찾기

        #create list to contain landmarks
        landmark_list = []

        # append (x, y) in landmark_list
        i = 10
        for p in landmarks.parts():
            landmark_list.append([p.x, p.y])
            i += 10 
            cv2.circle(image, (p.x, p.y), 2, (i, 255, 0), -1)
    
        import numpy.linalg as alg 

        #kinect setup
        cam_intrinsic = np.identity(3)
        cam_intrinsic[0, 0] = 1144
        cam_intrinsic[1, 1] = 1147
        cam_intrinsic[0, 2] = 1920/2
        cam_intrinsic[1, 2] = 1080/2

        pj = np.array(landmark_list).astype(np.float64)
        tvec = np.zeros(3)
        rvec = np.identity(3)
        _, rvec, tvec = cv2.solvePnP(v.reshape(v.shape[0], v.shape[1], 1), pj.reshape(pj.shape[0], pj.shape[1], 1), cam_intrinsic, np.zeros([5, 1]))
        
        print(tvec)
        print(rvec)
        kx = rvec[0]
        ky = rvec[1]
        kz = rvec[2]
        theta = np.linalg.norm(rvec)
        K = np.array([[0, -kz, ky],[kz, 0, -kx],[-ky, kx,0]])
        rot = np.identity(3) + np.sin(theta)*K + (1-np.cos(theta))*(K.dot(K))
        cmat = np.zeros([3,4])
        cmat[:, :3] = rot
        cmat[:,3] = tvec.reshape(-1)


        # pj[:,0] = pj[:,0] / 1920 
        # pj[:,1] = pj[:,1] / 1080 
        # pjmat = alg.lstsq(v, pj)[0]
        

        # flat_blendshape = blendshapes

        # flat_blendshape=np.concatenate([flat_blendshape, np.ones([flat_blendshape.shape[0], flat_blendshape.shape[1],1])], axis=-1)
        # flat_blendshape = np.einsum("ij,ljk->lik", cmat, np.transpose(flat_blendshape, axes=[0,2,1])) 
        # flat_blendshape = np.transpose(flat_blendshape,axes=[0,2,1])
        # flat_blendshape = flat_blendshape[..., :-1].reshape(flat_blendshape.shape[0], -1)

        # A = flat_blendshape.T
        ref_marker = cv2.projectPoints(v, rvec=rvec, tvec=tvec, cameraMatrix=cam_intrinsic, distCoeffs=np.zeros(5))[0]
        ref_marker = np.squeeze(ref_marker)

        # np.copy needed, because it's memory not contigous.(fuq...)
        flat_blendshape = np.zeros_like(blendshapes)[..., :-1]
        for i in range(flat_blendshape.shape[0]):
            proj_blendshape = cv2.projectPoints(np.copy(blendshapes[i]), rvec=rvec, tvec=tvec, cameraMatrix=cam_intrinsic, distCoeffs=np.zeros(5))[0]
            proj_blendshape = proj_blendshape - ref_marker[:,None,:] 
            flat_blendshape[i, ...] = np.transpose(proj_blendshape, [1,0,2])
        print(flat_blendshape)
        print(flat_blendshape.shape)
        A = flat_blendshape.reshape(flat_blendshape.shape[0], -1).T

        #set up 
        alpha = 0.01
        mu = 0.001
        ATA = A.T @ A
        t = 1.0
        w0 = np.zeros(A.shape[-1])


        #define marekr
        delta_pj = pj - ref_marker
        marker = delta_pj
        marker = marker.reshape(-1,1)
        
        S = 2*(ATA + ( alpha + mu ) * np.identity(ATA.shape[0]))
        q = np.squeeze(-2*A.T @ marker + alpha*w0.reshape(-1,1))
        G = np.vstack([np.identity(A.shape[-1]), -np.identity(A.shape[-1])])
        h = np.hstack( [t*np.ones_like(w0), np.zeros_like(w0)])

        res = solve_qp(P = S, q=q, G=G, h=h,A=None, b=None, lb = np.zeros_like(w0), ub=np.ones_like(w0), solver="cvxopt")
        print("res ", res)
        w0 = res
        result = v_all + (full_blendshape.reshape(full_blendshape.shape[0], -1).T @ res).reshape(v_all.shape)

        # ret_p = cv2.projectPoints(v_all, rvec=rvec, tvec=tvec, cameraMatrix=cam_intrinsic, distCoeffs=np.zeros(5))[0]
        ret_p = cv2.projectPoints(result, rvec=rvec, tvec=tvec, cameraMatrix=cam_intrinsic, distCoeffs=np.zeros(5))[0]
        # print(ret_p)
        for i in ret_p:
            i=i[0]
            cv2.circle(image, (int(i[0]), int(i[1])), 2, (255,0,0), -1)
    

    cv2.imshow('result', image)
    # Display color and depth data
    # cv2.imshow('color', color_img)
    # cv2.imshow('depth', depth_map_to_image(depth_map))
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    

kinect.disconnect()