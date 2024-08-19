/*
 * Linjie Luo - Princeton University
 * Chongyang Ma - Weta Digital
 */

#ifndef __ORIENT_MAP_HPP__
#define __ORIENT_MAP_HPP__

#include "Im.hpp"
#include <opencv2/opencv.hpp>

class OrientMap
{
public:
  OrientMap() : _image() {}

  OrientMap(int w, int h) : _image(w, h) {}

  OrientMap(const char* fileName)
  {
    read(fileName);
  }
    
    void read(Im& image)
    {
        _image = image;
        //for (int i = 0; i < _image.size(); i++) {
        //    vec c = _image[i] * 2.0f - vec(1, 1, 0);
        //    _image[i] = (c[2] > 0.0f ? c : vec());
        //}
        computeGradientMap();
    }
    
  const Im& get() const { return _image; }
  Im& get() { return _image; }

  Im getImage() const
  {
    Im im(_image.w, _image.h);
    for (int i = 0; i < _image.size(); i++) {
      im[i] = vec();
      if (_image[i][2] > 0.0f) {
	im[i][0] = _image[i][0] * 0.5f + 0.5f; 
	im[i][1] = _image[i][1] * 0.5f + 0.5f;
	im[i][2] = 1.0f;
      }
    }
    return im;
  }

  inline void doubleAngleToAngle(
				 float cos2A, float sin2A, float& cosA, float& sinA) const
  {
    cosA = sqrt(0.5f + cos2A * 0.5f);
    sinA = sqrt(0.5f - cos2A * 0.5f) * sgn(sin2A);
  }

  inline void angleToDoubleAngle(
				 float cosA, float sinA, float& cos2A, float& sin2A) const
  {
    cos2A = sqr(cosA) - sqr(sinA);
    sin2A = 2.0f * sinA * cosA;
  }

    vec2 get_samples_value(std::vector<vec2> samples)
    {
        vec2 avg_o(0,0);
        for(int i=0;i<samples.size();i++)
        {
            float x = 0.5*(samples[i][0]-0.5);
            float y = samples[i][1];
            float xx = x*x-y*y;
            float yy = 2*x*y;

            avg_o=avg_o+vec2(xx,yy);
        }

        avg_o[0]=avg_o[0]/samples.size();
        avg_o[1]=avg_o[1]/samples.size();

        float x = avg_o[0]/sqrt(avg_o[0]*avg_o[0]+avg_o[1]*avg_o[1]);
        float y = avg_o[1]/sqrt(avg_o[0]*avg_o[0]+avg_o[1]*avg_o[1]);

        x=x+1.0;
        y=y+0.0;
        float norm = sqrt(x*x+y*y);
        x= x/norm;
        y=y/norm;

        if(y<0)
        {
            x=-x;
            y=-y;
        }

        return vec2(x*0.5+0.5, y);

    }

  void diffuse(float lowConfThresh = 0.01f, int numIters = 40) {
      float lowConf2 = sqr(lowConfThresh);
      cout << "\nLowConf: " << lowConfThresh << "\n";    // 0.8

      int diffuse_num = 0;
      for (int i = 0; i < _image.size(); i++) {
          vec & c = _image[i];       
          if (c[2] < lowConfThresh) //if (c[2] > 0.0f && sqr(c[0]) + sqr(c[1]) < lowConf2)
          {
              c[2] = 0.5f; // mark as to be diffused
              diffuse_num = diffuse_num + 1;
          } else {
              c[2] = 1.0f;
          }
      }

      for (int iter = 0; iter <= numIters; iter++) {
          int diffuse_num2=0;
          for (int y = 0; y < _image.h; y++)
              for (int x = 0; x < _image.w; x++) {

                  // high confident pixels
                  if (_image(x, y)[2] > 0.5f) {
                      continue;
                  }
                  // background pixels
                  if (_image(x, y)[2] <= 0.0f) {
                      continue;
                  }
                  // boundary pixels
                  if (x <= 0 || x >= _image.w - 1 ||
                      y <= 0 || y >= _image.h - 1)
                      continue;

                  /*
                  std::vector<vec2> samples;
                  samples.push_back(vec2(_image(x, y - 1)[0], _image(x, y - 1)[1]));
                  samples.push_back(vec2(_image(x-1, y )[0], _image(x-1, y )[1]));
                  samples.push_back(vec2(_image(x, y )[0], _image(x, y )[1]));
                  samples.push_back(vec2(_image(x, y +1)[0], _image(x, y +1)[1]));
                  samples.push_back(vec2(_image(x+1, y )[0], _image(x+1, y )[1]));

                  vec2 c = get_samples_value(samples);
                  */
                  vec c = (_image(x, y - 1) +
                           _image(x - 1, y) +
                           _image(x, y) +
                           _image(x + 1, y) +
                           _image(x, y + 1)) / 5.0f;    // 4-connected neighbors weighted add
                  _image(x, y)[0] = c[0];
                  _image(x, y)[1] = c[1];
                  diffuse_num2++;

              }
      }
  }

  bool isForeground(int x, int y) const
  {
    if (0 <= x && x < _image.w && 0 <= y && y < _image.h &&
						_image(x, y)[2] > 0.0f) {
      return true;
    }
    return false;
  }

  bool read(const char* fileName)
  {
    if (!_image.read(fileName))
      return false;
    //for (int i = 0; i < _image.size(); i++) {
    //  vec c = _image[i] * 2.0f - vec(1, 1, 0);
   //   _image[i] = (c[2] > 0.0f ? c : vec());
    //}
    computeGradientMap();
    return true;
  }

    //added by Yi
    void write_orient_for_HairNet(const char* fileName) const
    {
        cv::Mat im(_image.h, _image.w, CV_32FC3);
        for (int i = 0; i < _image.size(); i++)
        {

            if (_image[i][2] > 0.0f) {

                //supose x is [0], y is [1]
                float x = _image[i][0];
                float y = _image[i][1];
                //cout<<y<<" ";
                im.at<cv::Vec3f>(i/_image.w, i% _image.w) [0] = x;//x*5.0;
                im.at<cv::Vec3f>(i/_image.w, i% _image.w) [1] = y;//y*0.5f+0.5f;
                im.at<cv::Vec3f>(i/_image.w, i% _image.w) [2] = 1.0f;

            }
        }
        cv::imwrite(fileName, im);
        // added by chenghong
        // std::cout << "Write: " << fileName << std::endl;
    }
    void write(const char* fileName) const
    {
        Im im(_image.w, _image.h);
        for (int i = 0; i < _image.size(); i++) {
            im[i] = vec();
            if (_image[i][2] > 0.0f) {
                im[i][0] = _image[i][0] * 0.5f + 0.5f;
                im[i][1] = _image[i][1] * 0.5f + 0.5f;
                im[i][2] = 0.5f;
            }
        }
        im.write(fileName);
  }

  void writeOrient(const char* fileName) const
  {
    Im im(_image.w, _image.h);
    for (int i = 0; i < _image.size(); i++) {
      im[i] = vec();
      if (_image[i][2] > 0.0f) {
	float conf = sqrt(sqr(_image[i][0]) + sqr(_image[i][1]));
	if (conf > 0.01f) {
	  im[i][0] = _image[i][0] / conf * 0.5f + 0.5f; 
	  im[i][1] = _image[i][1] / conf * 0.5f + 0.5f;
	  im[i][2] = 0.5f;
	} else {
	  im[i] = Color(0.5f);
	}
      }
    }
    im.write(fileName);
  }

  void writeHSV(const char* fileName) const
  {
    Im im(_image.w, _image.h);
    for (int i = 0; i < _image.size(); i++) {
      const vec& c = _image[i];
      if (c[2] > 0.0f) {
	float conf = sqrt(sqr(c[0]) + sqr(c[1]));
	if (conf > 0.01f) {
	  float a = atan2(c[1] / conf, c[0] / conf);
	  im[i] = Color::hsv(a, conf, conf * 0.5f + 0.5f);
	} else {
	  im[i] = Color(0.5f);
	}
      }
    }
    im.write(fileName);
  }

  void writeConfidence(const char* fileName) const
  {
      cv::Mat im(_image.h, _image.w, CV_32FC3);
      // std::string txt_filename=std::string(fileName);
      // txt_filename=txt_filename.substr(0,txt_filename.length()-4)+".txt";
      // std::ofstream ofile(txt_filename);
      // ofile<<_image.h<<" "<<_image.w<<std::endl;


      for (int i = 0; i < _image.size(); i++) {

          float cof=0;
          // ofile<<_image[i][0]<<" "<<_image[i][1]<<" "<<_image[i][2]<<std::endl;
          // cout<<_image[i][2]<<endl;
          if (_image[i][2] > 0.5f)
              cof = 0;//_image[i][2];//sqrt(sqr(_image[i][0]) + sqr(_image[i][1]));
          else
              cof = 255;

          //cout<<y<<" ";
          im.at<cv::Vec3f>(i / _image.w, i % _image.w) = cv::Vec3f(cof,cof,cof);


      }
      cv::imwrite(fileName, im);

      /*
    Im im(_image.w, _image.h);
    for (int i = 0; i < _image.size(); i++) {
      const vec& c = _image[i];
      if (c[2] > 0.0f)
	im[i] = Color(sqrt(sqr(c[0]) + sqr(c[1])));
      else
	im[i] = Color(0.0f);
    }
    im.write(fileName);
      */
  }

  void computeGradientMap()
  {
    int w = _image.w, h = _image.h;
    _grad.resize(w, h);
    for (int y = 0; y < h; y++)
      for (int x = 0; x < w; x++) {
	vec c0, c1;
	float l0, l1;
	c0 = _image(x - (x > 0), y);
	c1 = _image(x + (x<w-1), y);
	l0 = sqrt(sqr(c0[0]) + sqr(c0[1]));
	l1 = sqrt(sqr(c1[0]) + sqr(c1[1]));
	_grad(x, y)[0] =
	  (c0[2] > 0.0f && c1[2] > 0.0f ? (l1 - l0) * 0.5f : 0.0f);
	c0 = _image(x, y - (y > 0));
	c1 = _image(x, y + (y<h-1));
	l0 = sqrt(sqr(c0[0]) + sqr(c0[1]));
	l1 = sqrt(sqr(c1[0]) + sqr(c1[1]));
	_grad(x, y)[1] =
	  (c0[2] > 0.0f && c1[2] > 0.0f ? (l1 - l0) * 0.5f : 0.0f);
      }
  }

  // return orientation consistency energy
  inline float compare(
		       const vec2& point, const vec2& orient) const
  {
    vec2 thisOrient;
    float conf = sample(point, thisOrient, &orient);
    return min(dist(orient, thisOrient * conf), 1.0f);
  }


  inline float sample(
		      const vec2& point, vec2& orient, const vec2* refOrient = NULL) const
  {
    vec c = _image.lerp(point[0], point[1]);
    if (c[2] == 0.0f) return -1.0f;
    float conf = sqrt(sqr(c[0]) + sqr(c[1]));
    if (conf == 0.0f) return -1.0f;
    doubleAngleToAngle(c[0] / conf, c[1] / conf, orient[0], orient[1]);
    if (refOrient && (orient DOT *refOrient) < 0) orient = -orient;
    return conf;
  }

  inline vec2 sampleGradient(const vec2& point) const
  {
    vec g = _grad.lerp(point[0], point[1]);
    return vec2(g[0], g[1]);
  }

  void getRidgePoints(
		      vector<vec2>& points, float minConfidence, float minContrast) const
  {
    points.clear();
    Im nmax(_image.w, _image.h);
    for (int y = 0; y < _image.h; y++)
      for (int x = 0; x < _image.w; x++) {
	vec2 p(x, y);
	vec2 orient;
	float f1 = sample(p, orient);
	if (f1 < minConfidence)
	  continue;
	vec2 normal(orient[1], -orient[0]);
	float f0 = sample(p - normal, orient);
	float f2 = sample(p + normal, orient);
	float contrast = (f1 - max(f0, f2)) / f1;
	if (contrast > minContrast)
	  points.push_back(
			   snapToRidge(p - normal, f0, p, f1, p + normal, f2));
      }
  }

  inline vec2 snapToRidge(const vec2& point) const
  {
    float e0, e1, e2;
    vec2 orient;
    e1 = sample(point, orient);
    vec2 normal(orient[1], -orient[0]);
    e0 = sample(point - normal, orient);
    e2 = sample(point + normal, orient);
    return snapToRidge(point - normal, e0, point, e1, point + normal, e2);
  }

  inline vec2 snapToRidge(
			  const vec2& point0, float mag0,
			  const vec2& point1, float mag1,
			  const vec2& point2, float mag2) const
  {
    float E[3] = {mag0, mag1, mag2};
    float f, e;
    refineSubpixel(E, 3, 1, f, e);
    return (f < 1.0f ?
	    point0 * (1.0f - f) + point1 * f:
	    point1 * (2.0f - f) + point2 * (f - 1.0f));
  }

  int refineSubpixel(
		     const float* E, int n, int idep, 
		     float& depth, float& energy) const
  {
    float e0 = 0.0f, e1 = 0.0f, e2 = 0.0f;
    int move = 0;
    while (idep > 0 && idep < n - 1) {
      e0 = E[idep-1]; e1 = E[idep]; e2 = E[idep+1];
      if (e0 < e1 && e1 < e2) {
	depth = float(idep) + 0.5f;
	energy = e1;
	move = 1;
	break;
	// e0 = e1; e1 = e2; 
	// e2 = E[min(idep+2, n-1)]; // shift right
	// idep++; move++;
      } else if (e0 > e1 && e1 > e2) {
	depth = float(idep) - 0.5f;
	energy = e1;
	move = -1;
	break;
	// e2 = e1; e1 = e0; 
	// e0 = E[max(idep-2, 0)]; // shift left
	// idep--; move--;
      } else {
	float d2E = e0 + e2 - 2.0f * e1;
	depth = (d2E != 0.0f ? idep + 0.5f * (e0 - e2) / d2E : idep);
	energy = e1;
	break;
      }
    }
    return move;
  }

protected:
  Im _image;
  Im _grad;
  vector<vec2> _ridge;
  //kdtree2f* _kd;
};

#endif
