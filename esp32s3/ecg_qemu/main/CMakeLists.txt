idf_component_register(
    SRCS "main.cpp" "model_fp32.c"
    INCLUDE_DIRS "."
    EMBED_FILES "b_int16.espdl"
                "c_w8a16.espdl"
                "c3.espdl"
                "c1.espdl"
)
