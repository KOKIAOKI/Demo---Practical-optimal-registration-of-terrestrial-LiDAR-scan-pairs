#include <iostream>
#include <time.h>
#include <fstream>
#include <string>
#include <ctime>   // std::time
#include <cstdlib> // std::rand, std::srand

#include "../include/util/result_csv.hpp"
#include "../include/util/tool.hpp"

// external libraries
#include <pcl/io/auto_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/icp.h>
#include <pcl/common/transforms.h>
// source files
#include "../include/util/inputGen.h"
#include "../include/globReg4D/globReg4D.h"
#include "../include/globReg4D/LM.h"
#include "../include/globReg4D/GameTheoryAlbrarelli.h"
#include <pcl/registration/ia_kfpcs.h>
#include <chrono>
//! uncomment if you have installed and want to use S4PCS (please reset to the directory of the installed library in your computer)
// #include "../S4PCS/Super4PCS/build/install/include/pcl/registration/super4pcs.h"

using namespace std;
using namespace GenIn;
using namespace GLOBREG;

Eigen::Matrix4d optimization(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudS, pcl::PointCloud<pcl::PointXYZ>::Ptr issT, pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhT, double rT, double inlTh, Vector3 mCentT, int maxCorr, int testMethod)
{
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> colorS(cloudS, 255, 0, 0);

    // voxel grid filter (VGF)
    cout << "performing voxel grid sampling with grid size = " << inlTh << endl;
    VGF(cloudS, cloudS, inlTh);

    // extract ISS
    pcl::PointCloud<pcl::PointXYZ>::Ptr issS(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointIndicesPtr issIdxS(new pcl::PointIndices);
    cout << "extracting ISS keypoints..." << inlTh << endl;
    ISSExt(cloudS, issS, issIdxS, inlTh);
    cout << "size of issS = " << issS->size() << endl;

    // translating the center of both point clouds to the origin
    Vector3 centS(0, 0, 0);
    double rS;
    CentAndRComp(issS, centS, rS);
    Vector3 mCentS(-centS.x, -centS.y, -centS.z);
    transPC(issS, mCentS);

    // compute matches if needed
    // compute normal
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhS(new pcl::PointCloud<pcl::FPFHSignature33>());
    vector<corrTab> corr;
    vector<int> corrNOS, corrNOT;

    if (testMethod >= 1 && testMethod <= 5)
    {
        // compute fpfh
        cout << "computing fpfh..." << endl;
        FPFHComp(cloudS, inlTh, issIdxS, fpfhS);
        // match features
        cout << "matching correspodences..." << endl;
        corrComp(fpfhS, fpfhT, corr, maxCorr, corrNOS, corrNOT);
        cout << "NO. corr = " << corr.size() << endl;
    }

    // start optimization
    Transform3 result;

    if (testMethod == 1)
    {
        cout << "running FMA+BnB..." << endl;
        globReg4D(issS, issT, corr, corrNOS, corrNOT, inlTh, rS + rT, result);
    }
    else if (testMethod == 2)
    {
        cout << "running BnB..." << endl;
        globReg4D(issS, issT, corr, corrNOS, corrNOT, inlTh, rS + rT, result, false);
    }
    else if (testMethod == 3)
    {
        cout << "running RANSAC..." << endl;
        RANSAC(issS, issT, corr, inlTh, result);
    }
    else if (testMethod == 4)
    {
        cout << "running the Lifting Method (LM)..." << endl;
        CApp regFun;
        regFun.LM(issS, issT, corr, inlTh, result);
    }
    else if (testMethod == 5)
    {
        cout << "running the Game Theory Approach (GTA)..." << endl;
        GameTheoryAlbrarelli regFun;
        regFun.GTReg(issS, issT, corr, inlTh, result);
    }
    else if (testMethod == 6)
    {
        cout << "running K4PCS..." << endl;
        pcl::registration::KFPCSInitialAlignment<pcl::PointXYZ, pcl::PointXYZ> kfpcs;
        pcl::PointCloud<pcl::PointXYZ> final;
        kfpcs.setInputSource(issS);
        kfpcs.setInputTarget(issT);
        kfpcs.setNumberOfThreads(1);
        kfpcs.setDelta(inlTh, false);
        kfpcs.setScoreThreshold(0.001);
        kfpcs.align(final);
        Eigen::Matrix<float, 4, 4> a = kfpcs.getFinalTransformation();
        for (size_t i = 0; i < 4; i++)
        {
            for (size_t j = 0; j < 4; j++)
            {
                result.x[i + j * 4] = (double)a(i, j);
            }
        }
    }

    result = TransMatCompute(result, mCentS, mCentT);

    Eigen::Matrix4d transform;
    for (size_t i = 0; i < 4; i++)
    {
        for (size_t j = 0; j < 4; j++)
        {
            transform(i, j) = result.x[i + 4 * j];
        }
    }
    return transform;
}

void FPADependenceTest(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudS, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudT,
                       double inlTh, int maxCorr, int testNO, vector<int> &sizeResult)
{
    if (inlTh <= 0 || maxCorr <= 0)
    {
        cout << "inlier threshold or maximum correspondence number must > 0" << endl;
        return;
    }

    // voxel grid filter (VGF)
    cout << "performing voxel grid sampling with grid size = " << inlTh << endl;
    VGF(cloudS, cloudS, inlTh);
    VGF(cloudT, cloudT, inlTh);

    // extract ISS
    pcl::PointCloud<pcl::PointXYZ>::Ptr issS(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointIndicesPtr issIdxS(new pcl::PointIndices);
    pcl::PointCloud<pcl::PointXYZ>::Ptr issT(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointIndicesPtr issIdxT(new pcl::PointIndices);
    cout << "extracting ISS keypoints..." << inlTh << endl;
    ISSExt(cloudS, issS, issIdxS, inlTh);
    ISSExt(cloudT, issT, issIdxT, inlTh);
    cout << "size of issS = " << issS->size() << "; size of issT = " << issT->size() << endl;

    // translating the center of both point clouds to the origin
    Vector3 centS(0, 0, 0), centT(0, 0, 0);
    double rS, rT;
    CentAndRComp(issS, centS, rS);
    CentAndRComp(issT, centT, rT);
    Vector3 mCentS(-centS.x, -centS.y, -centS.z);
    Vector3 mCentT(-centT.x, -centT.y, -centT.z);
    transPC(issS, mCentS);
    transPC(issT, mCentT);

    // compute matches if needed
    // compute normal
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhS(new pcl::PointCloud<pcl::FPFHSignature33>());
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhT(new pcl::PointCloud<pcl::FPFHSignature33>());
    vector<corrTab> corr;
    vector<int> corrNOS, corrNOT;

    // compute fpfh
    cout << "computing fpfh..." << endl;
    FPFHComp(cloudS, inlTh, issIdxS, fpfhS);
    FPFHComp(cloudT, inlTh, issIdxT, fpfhT);
    // match features
    cout << "matching correspodences..." << endl;
    corrComp(fpfhS, fpfhT, corr, maxCorr, corrNOS, corrNOT);
    cout << "NO. corr = " << corr.size() << endl;

    sizeResult[0] = corr.size();

    // convert data format
    Matrix3X matrixS;
    DataTrans(issS, matrixS);
    Matrix3X matrixT;
    DataTrans(issT, matrixT);

    Transform3 result;

    vector<int> sizeAfFMP(testNO);
    std::srand(unsigned(std::time(0)));

    // start testing
    for (size_t i = 0; i < testNO; i++)
    {
        // re-initialize and shuffle the correspondences
        vector<corrTab> corrCopy(corr);
        // using built-in random generator:
        std::random_shuffle(corrCopy.begin(), corrCopy.end());

        vector<int> corrNOSCopy(corrNOS), corrNOTCopy(corrNOT);

        cout << "NO correspondences before FMP = " << corrCopy.size() << endl;
        FPA(matrixS, matrixT, corrCopy, corrNOSCopy, corrNOTCopy, inlTh, result);
        cout << "NO correspondences after FMP = " << corrCopy.size() << endl;
        sizeResult[i + 1] = corrCopy.size();
        sizeAfFMP[i] = corrCopy.size();
    }

    // compute average
    int sum = std::accumulate(sizeAfFMP.begin(), sizeAfFMP.end(), 0);
    float mean = sum / ((float)testNO);
    float sd = 0;
    for (size_t i = 0; i < testNO; i++)
    {
        sd += (sizeAfFMP[i] - mean) * (sizeAfFMP[i] - mean);
    }
    int max = *max_element(sizeAfFMP.begin(), sizeAfFMP.end());
    int min = *min_element(sizeAfFMP.begin(), sizeAfFMP.end());
    sd = sqrt(sd) / (float)testNO;
    cout << "mean = " << mean << "; sd = " << sd << "; max = " << max << "; min = " << min;
}

void saveFMPDependence(string fname, std::vector<int> outVec)
{
    std::ofstream output_file(fname);
    for (const auto &e : outVec)
        output_file << e << "\n";
}

int main(int argc, char *argv[])
{
    // INPUT:
    // 1. path to the source point cloud
    string source_folder = argv[1];
    // 2. path to the target point cloud
    string target_folder = argv[2];
    // 3. ground truth file
    string gt_file_path = argv[3];
    // 4. output result file
    string output_folder_path = argv[4];
    // 5. inlier threshold (0.2~0.3)
    double inlTh = atof(argv[5]);
    // 6. maximum correspondence number (default 10)
    int maxCorr = atoi(argv[6]);
    // 7. method u want to run (from 1 to 6, default 1)
    //    1. FMP+BnB; 2. BnB; 3. LM; 4. GTA; 5.RANSAC 6. K4PCS; 7. S4PCS
    int testMethod = atoi(argv[7]);

    if (inlTh <= 0 || maxCorr <= 0)
    {
        cout << "inlier threshold or maximum correspondence number must > 0" << endl;
        return 1;
    }
    if (testMethod < 1 || testMethod > 7)
    {
        cout << "wrong testing method, use numbers from 1 to 7 to specify" << endl;
        return 1;
    }

    // Create output folder with date and create result csv
    std::string date = create_date();
    std::string pcd_save_folder_path = output_folder_path + "/" + date;
    std::filesystem::create_directory(pcd_save_folder_path);
    ResultCsv result_csv(pcd_save_folder_path, gt_file_path);

    // read pcd files
    auto src_cloud_files = find_point_cloud_files(source_folder);
    auto tar_cloud_files = find_point_cloud_files(target_folder);

    // sort src files
    if (can_convert_to_double(src_cloud_files))
    {
        std::sort(src_cloud_files.begin(), src_cloud_files.end(), [](const std::string &a, const std::string &b)
                  { return std::stod(std::filesystem::path(a).stem().string()) < std::stod(std::filesystem::path(b).stem().string()); });
    }

    // pre process target cloud
    const auto cloudT = create_tar_cloud(tar_cloud_files);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> colorT(cloudT, 255, 255, 0);
    VGF(cloudT, cloudT, inlTh);

    pcl::PointCloud<pcl::PointXYZ>::Ptr issT(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointIndicesPtr issIdxT(new pcl::PointIndices);
    ISSExt(cloudT, issT, issIdxT, inlTh);
    std::cout << "size of issT = " << issT->size() << std::endl;

    Vector3 centT(0, 0, 0);
    double rT;
    CentAndRComp(issT, centT, rT);
    Vector3 mCentT(-centT.x, -centT.y, -centT.z);
    transPC(issT, mCentT);

    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhT(new pcl::PointCloud<pcl::FPFHSignature33>());
    FPFHComp(cloudT, inlTh, issIdxT, fpfhT);

    for (const auto &src_cloud_file : src_cloud_files)
    {
        // read point clouds
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudS(new pcl::PointCloud<pcl::PointXYZ>);
        if (pcl::io::load<pcl::PointXYZ>(src_cloud_file, *cloudS) == -1) //* load the file
        {
            cout << "Couldn't read file: " << src_cloud_file << endl;
            return (-1);
        }
        std::cout << "size of cloudS = " << cloudS->size() << std::endl;

        const auto start_time = std::chrono::system_clock::now();

        const Eigen::Matrix4d result_T = optimization(cloudS, issT, fpfhT, rT, inlTh, mCentT, maxCorr, testMethod);

        const auto end_time = std::chrono::system_clock::now();
        const double elapsed_time_msec = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() / 1e6;

        result_csv.write(
            std::filesystem::path(src_cloud_file).stem().string(),
            elapsed_time_msec,
            result_T,
            Eigen::Matrix4d::Identity());

        pcl::transformPointCloud(*cloudS, *cloudS, result_T);
        pcl::io::savePCDFileBinary(pcd_save_folder_path + "/" + std::filesystem::path(src_cloud_file).stem().string() + ".pcd", *cloudS);
    }
    return 0;
}
