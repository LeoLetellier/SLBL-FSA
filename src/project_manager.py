import numpy as np

from LandslideAreanSLBL import *
from SlblToolKit import *

COMMENT_CHARACTER = '#'

# KEYWORDS
GLOBAL = ['NAME', 'DEM_PATH']
SLIDE_PATH = ['METHOD', 'PATH']
SLIDE_RANGE = ['METHOD', 'LEFT_LIM', 'RIGHT_LIM', 'TOL']
SLIDE_COMBINE = ['METHOD', 'SURF1', 'SURF2']
SLIDE_SUB = ['METHOD', 'LEFT_LIM', 'RIGHT_LIM', 'TOL', 'SURF_MASTER']
SLIDE_POINT = ['METHOD', 'PNT1_X', 'PNT1_Z', 'PNT2_X', 'PNT2_Z', 'TOL']
SLIDE = [SLIDE_PATH, SLIDE_RANGE, SLIDE_COMBINE, SLIDE_SUB, SLIDE_POINT]
DISP_VEC = ['N_VEC', 'X_MIN', 'X_MAX']
DISP_MODEL = ['SLIDES', 'COEFFS', 'METHOD', 'DISP_PATH', 'ALPHA', 'THETA', 'DELTA']


class ProjectManager:
    def __init__(self):
        self.name = ''
        self.path = ''
        self.global_content = dict()
        self.slides_content = [dict(), dict(), dict(), dict(), dict(), dict()]
        self.disp_content = dict()
        self.model_content = dict()

    def save_project(self, area: LandslideArea, path: str):
        self.path = path
        self.name = self.path[-(self.path[::-1].find('/')):-(self.path[::-1].find('.') + 1)]

        output = 'GLOBAL\n'
        dem_path = None
        if area.dem_path is not None:
            dem_path = area.dem_path
        output += self.save_content(GLOBAL, [self.name, dem_path])
        output += '\n'

        for k in range(6):
            sld = area.slides['slide' + str(k + 1)]
            if sld.method is not None:
                output += 'SLIDE' + str(k + 1) + '\n'
                output += self.save_slide(area, sld)
                output += '\n'

        mdl = area.model
        if mdl.n_vec is not None:
            output += 'DISP_VEC\n'
            output += self.save_content(DISP_VEC, [mdl.n_vec, area.x[mdl.start_pnt], area.x[mdl.stop_pnt]])
            output += '\n'

            if area.dc.disp_model is not None:
                output += 'DISP_MODEL\n'
                output += self.save_content(DISP_MODEL, self.save_model(area))
                output += '\n'

        file_out = open(self.path, 'w')
        file_out.write(output)
        file_out.close()

    def save_content(self, keywords, content):
        if content is not None:
            if len(keywords) == len(content):
                output = ''
                for k in range(len(keywords)):
                    output = output + keywords[k] + ' ' + str(content[k]) + '\n'
                return output
        return None

    def save_slide(self, area: LandslideArea, slide: Slide):
        method = ['PATH', 'RANGE', 'COMBINE', 'SUB', 'POINT']
        id = method.index(slide.method)
        match id:
            case 0:
                content = [slide.method, slide.slide_path]
            case 1:
                content = [slide.method, area.x[slide.start_pnt], area.x[slide.stop_pnt], slide.slbl.c]
            case 2:
                content = [slide.method, slide.comb1[-1], slide.comb2[-1]]
            case 3:
                content = [slide.method, area.x[slide.start_pnt], area.x[slide.stop_pnt], slide.slbl.c, slide.master[-1]]
            case 4:
                content = [slide.method, slide.points[0][0], slide.points[0][1], slide.points[1][0], slide.points[1][1], slide.slbl.c]
            case _:
                return None
        return self.save_content(SLIDE[id], content)

    def save_model(self, area):
        slides = ''
        coeffs = ''
        for k in range(6):
            sld = area.slides['slide' + str(k + 1)]
            if sld.method is not None:
                if slides != '':
                    slides += ';'
                    coeffs += ';'
                slides += str(k + 1)
                coeffs += str(area.slides['slide' + str(k + 1)].disp_ratio)
        disp_path = 'None'
        if area.dc.disp_path is not None:
            disp_path = area.dc.disp_path
        return slides, coeffs, area.dc.method, disp_path, area.dc.alpha, area.dc.theta, area.dc.delta

    def load_project(self, area: LandslideArea, path: str):
        self.path = path
        self.name = self.path[-(self.path[::-1].find('/')):-(self.path[::-1].find('.') + 1)]

        file_in = open(self.path, 'r')
        input = file_in.read()
        file_in.close()
        self.read_input(input)

        self.dispatch_global(area)
        self.dispatch_slides(area)
        if len(self.disp_content) != 0:
            self.dispatch_disp(area)
            if len(self.model_content) != 0:
                self.dispatch_model(area)

    def read_input(self, input):
        lines = input.split('\n')
        send_in = []
        for line in lines:
            if len(line) != 0:
                if line[0] != COMMENT_CHARACTER:
                    send_in.append(line)
            else:
                if len(send_in) != 0:
                    self.handle_send_in(send_in)
                    send_in = []

    def handle_send_in(self, send_in):
        match send_in[0]:
            case 'GLOBAL':
                self.handler(self.global_content, send_in)
                return True
            case 'SLIDE1':
                self.handler(self.slides_content[0], send_in)
                return True
            case 'SLIDE2':
                self.handler(self.slides_content[1], send_in)
                return True
            case 'SLIDE3':
                self.handler(self.slides_content[2], send_in)
                return True
            case 'SLIDE4':
                self.handler(self.slides_content[3], send_in)
                return True
            case 'SLIDE5':
                self.handler(self.slides_content[4], send_in)
                return True
            case 'SLIDE6':
                self.handler(self.slides_content[5], send_in)
                return True
            case 'DISP_VEC':
                self.handler(self.disp_content, send_in)
                return True
            case 'DISP_MODEL':
                self.handler(self.model_content, send_in)
                return True
            case _:
                return False

    def handler(self, dic, send_in):
        for el in send_in[1:]:
            split = el.split()
            dic[split[0]] = split[1]

    def dispatch_global(self, area: LandslideArea):
        area.dem_path = self.global_content['DEM_PATH']
        area.load_dem(self.global_content['DEM_PATH'])

    def dispatch_slides(self, area: LandslideArea):
        for k in range(6):
            dic = self.slides_content[k]
            slide = area.slides['slide' + str(k + 1)]
            if len(dic) != 0:
                match dic['METHOD']:
                    case 'PATH':
                        slide.method = dic['METHOD']
                        slide.slide_path = dic['PATH']
                        slide.slbl.load(slide.slide_path)
                    case 'RANGE':
                        slide.method = dic['METHOD']
                        slide.start_pnt = meter_2_idx(float(dic['LEFT_LIM']), area.x)
                        slide.stop_pnt = meter_2_idx(float(dic['RIGHT_LIM']), area.x)
                        slide.slbl.c = float(dic['TOL'])
                        slide.range_2_slbl()
                    case 'COMBINE':
                        slide.method = dic['METHOD']
                        slide.comb1 = 'slide' + dic['SURF1']
                        slide.comb2 = 'slide' + dic['SURF2']
                        slide.combine_slbl(slide.comb1, slide.comb2)
                    case 'SUB':
                        slide.method = dic['METHOD']
                        slide.start_pnt = meter_2_idx(float(dic['LEFT_LIM']), area.x)
                        slide.stop_pnt = meter_2_idx(float(dic['RIGHT_LIM']), area.x)
                        slide.slbl.c = float(dic['TOL'])
                        slide.master = 'slide' + dic['SURF_MASTER']
                        slide.sub_slbl(slide.master)
                    case 'POINT':
                        slide.method = dic['METHOD']
                        slide.slbl.c = float(dic['TOL'])
                        slide.points = [[float(dic['PNT1_X']), float(dic['PNT1_Z'])], [float(dic['PNT2_X']), float(dic['PNT2_Z'])]]
                        slide.point_slbl(slide.points[0], slide.points[1])

    def dispatch_disp(self, area: LandslideArea):
        dic = self.disp_content
        area.model.n_vec = int(dic['N_VEC'])
        area.model.start_pnt = meter_2_idx(float(dic['X_MIN']), area.x)
        area.model.stop_pnt = meter_2_idx(float(dic['X_MAX']), area.x)
        for k in range(6):
            slide = area.slides['slide' + str(k + 1)]
            if len(self.slides_content[k]) != 0:
                area.model.slope_direction = np.sign((area.z[-1] - area.z[0]) / (area.x[-1] - area.x[0]))
                slide.block_disp.compute_disp_vectors()

    def dispatch_model(self, area: LandslideArea):
        dic = self.model_content
        area.model.corresponding_slides = ['slide' + nb for nb in dic['SLIDES'].split(';')]
        for k in range(6):
            slide = area.slides['slide' + str(k + 1)]
            if len(self.slides_content[k]) != 0:
                idx = dic['SLIDES'].split(';').index(str(k + 1))
                slide.disp_ratio = float(dic['COEFFS'].split(';')[idx])
        area.dc.method = dic['METHOD']
        if dic['DISP_PATH'] != 'None':
            area.dc.disp_path = dic['DISP_PATH']
            area.dc.alpha = float(dic['ALPHA'])
            area.dc.theta = float(dic['THETA'])
            area.dc.delta = float(dic['DELTA'])

            area.dc.load(area.dc.disp_path)
            match area.dc.method:
                case 'USER':
                    area.model.slides_2_model()
                    area.dc.proj_disp()
                case 'LEAST_SQUARE':
                    area.dc.model_ajust_ratio_mean_square()
