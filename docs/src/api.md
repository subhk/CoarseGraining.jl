# API Reference

This page lists the public API documented in the source.

```@index
Modules = [CoarseGraining]
Order   = [:type, :function, :macro]
```

```@autodocs
Modules = [CoarseGraining]
Order   = [:module, :type, :function, :macro]
Private = false
```

## IO Submodule

```@autodocs
Modules = [CoarseGraining.IO]
Order   = [:type, :function, :macro]
Private = false
```

## ModelIO Submodule

```@autodocs
Modules = [CoarseGraining.ModelIO]
Order   = [:type, :function, :macro]
Private = false
```

## MPIUtils Submodule

```@autodocs
Modules = [CoarseGraining.MPIUtils]
Order   = [:type, :function, :macro]
Private = false
```

## Internal API (non-exported)

The following lists non-exported names to ensure full coverage in `checkdocs = :all`.

```@autodocs
Modules = [CoarseGraining, CoarseGraining.IO, CoarseGraining.ModelIO, CoarseGraining.MPIUtils]
Order   = [:module, :type, :function, :macro]
Private = true
Public  = false
```
