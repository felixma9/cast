#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "StrainUtility.h"
#include "RFImageInfo.h"

namespace py = pybind11;

PYBIND11_MODULE(strain_utility, m) {
    m.doc() = "StrainUtility bindings for ultrasound strain elastography processing";

    // Binding for RFIMAGEINFO structure
    py::class_<RFIMAGEINFO>(m, "RFImageInfo")
        .def(py::init<>())
        .def_readwrite("nLineNum", &RFIMAGEINFO::nLineNum)
        .def_readwrite("nBlockNum", &RFIMAGEINFO::nBlockNum)
        .def_readwrite("nBlockLength", &RFIMAGEINFO::nBlockLength)
        .def_readwrite("nLineLength", &RFIMAGEINFO::nLineLength)
        .def_readwrite("nBlockSpacing", &RFIMAGEINFO::nBlockSpacing)
        .def_readwrite("nStartLine", &RFIMAGEINFO::nStartLine)
        .def_readwrite("nEndLine", &RFIMAGEINFO::nEndLine)
        .def_readwrite("nStartBlock", &RFIMAGEINFO::nStartBlock)
        .def_readwrite("nEndBlock", &RFIMAGEINFO::nEndBlock)
        .def_readwrite("fAmplify", &RFIMAGEINFO::fAmplify)
        .def_readwrite("uStrainLSE", &RFIMAGEINFO::uStrainLSE)
        .def_readwrite("bMedian", &RFIMAGEINFO::bMedian)
        .def_readwrite("bLateral", &RFIMAGEINFO::bLateral)
        .def_readwrite("bExtSearch", &RFIMAGEINFO::bExtSearch)
        .def_readwrite("nKernelSize", &RFIMAGEINFO::nKernelSize);

    // Binding for StrainUtility class
    py::class_<StrainUtility>(m, "StrainUtility")
        .def(py::init<>())
        
        // Correlation estimator methods
        .def("setCorrelationEstimator", &StrainUtility::setCorrelationEstimator,
             py::arg("correlationEstimator"),
             "Set correlation estimator type (0: non-normalized, 1: normalized, 2: SSD, 3: covariance)")
        .def("getCorrelationEstimator", &StrainUtility::getCorrelationEstimator,
             "Get current correlation estimator type")
        
        // Reset/initialization
        .def("Reset", &StrainUtility::Reset,
             py::arg("nLineNum"),
             py::arg("nBlockNum"),
             py::arg("bKalman") = false,
             "Initialize buffer for temporal filtering")
        .def("clearTemporalBuffer", &StrainUtility::clearTemporalBuffer,
             "Clear temporal filtering buffer")
        
        // Main TDPE processing function
        .def("TDPE", [](StrainUtility &s,
                        py::array_t<uint8_t> pre_buffer,
                        py::array_t<uint8_t> buffer,
                        py::array_t<float> axial_disp,
                        py::array_t<float> lateral_disp,
                        py::array_t<float> strain,
                        py::array_t<float> MS,
                        py::array_t<float> search_map,
                        RFIMAGEINFO &img_info) -> float {
            
            // Get raw pointers (with automatic bounds checking disabled for performance)
            auto pre_buf = pre_buffer.unchecked<1>();
            auto cur_buf = buffer.unchecked<1>();
            auto ax_disp = axial_disp.mutable_unchecked<1>();
            auto lat_disp = lateral_disp.mutable_unchecked<1>();
            auto str = strain.mutable_unchecked<1>();
            auto corr = MS.mutable_unchecked<1>();
            auto search = search_map.mutable_unchecked<1>();
            
            // Call the C++ TDPE function
            return s.TDPE(
                (BYTE*)pre_buf.data(0),
                (BYTE*)cur_buf.data(0),
                (float*)ax_disp.data(0),
                (float*)lat_disp.data(0),
                (float*)str.data(0),
                (float*)corr.data(0),
                (float*)search.data(0),
                &img_info
            );
        },
        py::arg("pre_buffer"),
        py::arg("buffer"),
        py::arg("axial_disp"),
        py::arg("lateral_disp"),
        py::arg("strain"),
        py::arg("MS"),
        py::arg("search_map"),
        py::arg("img_info"),
        "Time Domain Cross-correlation with Prior Estimates (TDPE)\n"
        "Performs motion tracking and strain estimation between RF frames\n"
        "\n"
        "Args:\n"
        "    pre_buffer: Previous RF frame (uint8 array)\n"
        "    buffer: Current RF frame (uint8 array)\n"
        "    axial_disp: Output axial displacement map (float array, modified in-place)\n"
        "    lateral_disp: Output lateral displacement map (float array, modified in-place)\n"
        "    strain: Output strain estimation map (float array, modified in-place)\n"
        "    MS: Output motion similarity/correlation map (float array, modified in-place)\n"
        "    search_map: Output search map (float array, modified in-place)\n"
        "    img_info: RFImageInfo structure with processing parameters\n"
        "\n"
        "Returns:\n"
        "    Average normalized cross-correlation (0-1)")
        
        // Properties
        .def_readwrite("threshold", &StrainUtility::threshold,
                      "Motion tracking threshold")
        .def_readwrite("neighbours", &StrainUtility::neighbours,
                      "Least square estimator neighbor count")
        .def_readwrite("averaging", &StrainUtility::averaging,
                      "Temporal averaging factor")
        .def_readwrite("minCorrelation", &StrainUtility::minCorrelation,
                      "Minimum correlation threshold")
        .def_readwrite("maxCorrelation", &StrainUtility::maxCorrelation,
                      "Maximum correlation threshold")
        .def_readwrite("signPreserve", &StrainUtility::signPreserve,
                      "Preserve sign of strain values")
        .def_readwrite("SOPVar", &StrainUtility::SOPVar,
                      "Sum of product variant (0: non-normalized, 1: normalized, 2: SSD, 3: covariance)");
}
