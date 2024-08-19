import os
import torch
import numpy as np
import open3d as o3d
import torch.nn as nn
from lib.mesh_util import load_obj_mesh

def load_obj_with_line_elements(mesh_path):
    vertex_data = []
    line_element_data = []
    
    file = open(mesh_path, 'r')

    for line in file:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue
        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
        elif values[0] == 'mtllib':
            continue
        elif values[0] == '0':
            continue
        elif values[0] == 'l':
            l = list(map(int, values[1:3]))
            line_element_data.append(l)
    vertices = np.array(vertex_data)
    line_elements = np.array(line_element_data)
    file.close()

    return vertices, line_elements

def save_obj_with_line_elements(mesh_path, verts, lines):
    file = open(mesh_path, 'w')
    
    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for l in lines:
        file.write('l %d %d \n' % (l[0]+1, l[1]+1))
    file.close()

def write_strand2obj(dst_path, strands):
    assert dst_path.endswith('obj'), "Error, invalid dst_path!"

    pc = []
    lines = []
    sline = 0

    for i in range(len(strands)):
        for j in range(strands.shape[1]):
            pc.append(strands[i][j])
            if j == strands.shape[1] - 1:
                continue     ## for the last node of a strand: not save index
            lines.append([sline + j, sline + j + 1])
        
        sline += len(strands[i])
    
    save_obj_with_line_elements(dst_path, pc, np.asarray(lines))



def get_minmax(vertices):
    bmin_x = min(vertices[:,0])
    bmin_y = min(vertices[:,1])
    bmin_z = min(vertices[:,2])

    bmax_x = max(vertices[:,0])
    bmax_y = max(vertices[:,1])
    bmax_z = max(vertices[:,2])

    return np.array([bmin_x, bmin_y, bmin_z]), np.array([bmax_x, bmax_y, bmax_z])

def hair_synthesis(net, cuda, root_tensor, image_tensor, calib_tensor, num_sample=10, hair_unit=0.006):
    #root:[3, 1024]
    num_strand = root_tensor.shape[2]
    hair_strands = torch.zeros(num_sample, 3, num_strand).to(device=cuda)
    
    #hair_unit = 0.0015  #s, xxs
    #hair_unit = 0.01  #XXL
    curr_node = root_tensor.squeeze()
    hair_strands[0] = curr_node
    for i in range(1,num_sample):
        curr_node_orien = net.query(curr_node.unsqueeze(0), calib_tensor).squeeze()
        # normalizer = nn.functional.normalize
        # curr_node_orien = normalizer(curr_node_orien, dim=0)
        hair_strands[i] = hair_strands[i-1] + hair_unit * curr_node_orien
        curr_node = hair_strands[i]

    return hair_strands.permute(2, 0, 1).cpu().detach().numpy()


def hair_synthesis_DSH(net, cuda, root_tensor, image_tensor, calib_tensor, num_sample=100, hair_unit=0.006, threshold=[60,150]):
    #growing algorithm in DeepSketchHair
    #root:[3, 1024]
    num_strand = root_tensor.shape[2]
    hair_strands = torch.zeros(num_sample, 3, num_strand).to(device=cuda)
    
    curr_node = root_tensor.squeeze()
    last_node_orien = 0
    hair_strands[0] = curr_node
    for i in range(1, num_sample):
        curr_node_orien = net.query(curr_node.unsqueeze(0), calib_tensor).squeeze()

        if i>1:
            len_cd = torch.norm(curr_node_orien,p=2,dim=0)
            len_pd = torch.norm(last_node_orien,p=2,dim=0)
            in_prod = torch.sum(curr_node_orien * last_node_orien, dim=0)
            theta = torch.acos( in_prod/ (len_cd*len_pd))*180/np.pi

            idx_big_theta = theta > threshold[1]
            idx_mid_theta = ((theta > threshold[0]).float() - idx_big_theta.float()).bool().unsqueeze(0) # 60 < theta < 150
            idx_stop = (idx_big_theta + torch.isnan(theta).float()).bool().unsqueeze(0)  # orien=0 or theta>150

            idx_stop = torch.cat((idx_stop,idx_stop,idx_stop),dim=0)
            idx_mid_theta = torch.cat((idx_mid_theta,idx_mid_theta,idx_mid_theta),dim=0)
            

            half_node_orien = (curr_node_orien + last_node_orien) / 2

            curr_node_orien = torch.where(idx_mid_theta, half_node_orien, curr_node_orien)
            curr_node_orien = torch.where(idx_stop, half_node_orien, curr_node_orien)

        normalizer = nn.functional.normalize
        curr_node_orien = normalizer(curr_node_orien, dim=0)
        hair_strands[i] = hair_strands[i-1] + hair_unit * curr_node_orien
        curr_node = hair_strands[i]
        last_node_orien = curr_node_orien

    return hair_strands.permute(2, 0, 1).cpu().detach().numpy()


def save_strands_ply(strands, outputpath):
    # strands: [1024,100,3]
    pc_all_valid = []
    lines = []
    sline = 0

    for i in range(strands.shape[0]):
        if np.dot(strands[i,0], strands[i,0])<0.001 or np.dot(strands[i,1], strands[i,1])<0.001:
            continue
        num_pt = 2
        pc_all_valid.append(strands[i][0])
        pc_all_valid.append(strands[i][1])
        lines.append([sline + 0, sline + 1])

        for j in range(2,strands.shape[1]):
            if np.dot(strands[i,j], strands[i,j])>0.001:
                pc_all_valid.append(strands[i][j])
                lines.append([sline + j-1, sline + j])
                num_pt += 1
            else:
                break
        sline += num_pt
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(np.asarray(pc_all_valid)), lines=o3d.utility.Vector2iVector(lines))

    output_dir = outputpath[:outputpath.rfind('/')]   
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if outputpath.endswith(".ply"):
        o3d.io.write_line_set(outputpath, line_set)
    else:
        save_strands_obj(line_set, outputpath)

def save_strands_obj(line_set, filepath):
    filepath = filepath[:filepath.rfind('/')]  + filepath[filepath.rfind('/'):-3] + 'obj'
    hair_file = open(filepath,'w')

    points = np.array(line_set.points)
    lines = np.array(line_set.lines)

    for i in range(points.shape[0]):
        hair_file.write('v {0:.06f} {1:.06f} {2:.06f}\n'.format(points[i][0], points[i][1], points[i][2]))

    for i in range(lines.shape[0]):
            hair_file.write('l {0} {1}\n'.format(lines[i][0]+1, lines[i][1]+1))
    
    hair_file.close()


def get_hair_root(filepath):
    root, _ = load_obj_mesh(filepath)
    return root.T

def export_hair_real(net, cuda, data, save_path, hair_root_path, opt=None):
    image_tensor = data['img'].to(device=cuda).unsqueeze(0)
    calib_tensor = data['calib'].to(device=cuda).unsqueeze(0)
    root_tensor = torch.from_numpy(get_hair_root(hair_root_path)).to(device=cuda).float().unsqueeze(0)

    net.filter(image_tensor)

    strands = hair_synthesis(net, cuda, root_tensor, image_tensor, calib_tensor, num_sample=13, hair_unit=0.014)
    # strands = hair_synthesis_DSH(net, cuda, root_tensor, image_tensor, calib_tensor, num_sample=6, hair_unit=0.014, threshold=[30, 60])    # 6 for fixed_mean_length
 
    save_strands_ply(strands, save_path)

def cal_eyeb_orien(strands):
    #convert hair strands to orientation field
    hair_orien = np.zeros((strands.shape[0], strands.shape[1], 3)) # number of strands x number of points on each strand x 3 dimension

    for n in range(strands.shape[0]):
        for i in range(strands.shape[1] - 1):
            hair_orien[n,i] = strands[n,i+1] - strands[n,i]
            norm_length = np.sqrt(np.dot(hair_orien[n,i], hair_orien[n,i]))
            if norm_length > 0:
                hair_orien[n,i] /= norm_length
        hair_orien[n, strands.shape[1] - 1] = hair_orien[n, strands.shape[1] - 2]

    return hair_orien