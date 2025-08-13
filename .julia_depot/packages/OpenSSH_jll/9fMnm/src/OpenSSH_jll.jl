# Use baremodule to shave off a few KB from the serialized `.ji` file
baremodule OpenSSH_jll
using Base
using Base: UUID
import JLLWrappers

JLLWrappers.@generate_main_file_header("OpenSSH")
JLLWrappers.@generate_main_file("OpenSSH", UUID("9bd350c2-7e96-507f-8002-3f2e150b4e1b"))
end  # module OpenSSH_jll
