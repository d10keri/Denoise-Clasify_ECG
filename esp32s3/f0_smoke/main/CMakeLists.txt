idf_component_register(
    SRCS "main.cpp"
    INCLUDE_DIRS "."
    EMBED_FILES "../../quant/b_int16.espdl"
                "../../quant/c_w8a16.espdl"
                "../../quant/c3.espdl"
                "../../quant/c1.espdl"
)
